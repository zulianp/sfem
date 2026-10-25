#include <type_traits>
#include <cstdint>
#include <cstdlib>
#include <string.h>
#include "vector_mass_d3_simplex_local.cuh"
#include "geometry_kernels.cuh"
#include "kernel_diagnostics.cuh"
#include "quad_tet_q4.cuh"
#include "tet4_q4.cuh"
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

static const KernelDiagnostics vector_mass_tet4_residual_esoa_diagnostics_data = {
  "vector_mass_tet4_residual_esoa",
  "TET4",
  3,
  4,
  4,
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
  1012,
  1464,
  0,
  3,
  10,
  64,
  4,
  1,
  12,
  0,
  12,
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

extern "C" const sfem::codegen::KernelDiagnostics *cu_vector_mass_tet4_residual_esoa_diagnostics(void) {
  return &sfem::codegen::vector_mass_tet4_residual_esoa_diagnostics_data;
}

namespace sfem {
namespace codegen {

static const KernelDiagnostics vector_mass_tet4_jacobian_u0_u0_diagnostics_data = {
  "vector_mass_tet4_jacobian_u0_u0",
  "TET4",
  3,
  4,
  4,
  16,
  2,
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
  1012,
  1464,
  0,
  3,
  10,
  64,
  4,
  1,
  0,
  12,
  12,
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

extern "C" const sfem::codegen::KernelDiagnostics *cu_vector_mass_tet4_jacobian_u0_u0_diagnostics(void) {
  return &sfem::codegen::vector_mass_tet4_jacobian_u0_u0_diagnostics_data;
}

namespace sfem {
namespace codegen {

static const KernelDiagnostics vector_mass_tet4_jacobian_u0_u1_diagnostics_data = {
  "vector_mass_tet4_jacobian_u0_u1",
  "TET4",
  3,
  4,
  4,
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
  1,
  0,
  1012,
  1464,
  0,
  0,
  10,
  64,
  4,
  0,
  0,
  0,
  12,
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

extern "C" const sfem::codegen::KernelDiagnostics *cu_vector_mass_tet4_jacobian_u0_u1_diagnostics(void) {
  return &sfem::codegen::vector_mass_tet4_jacobian_u0_u1_diagnostics_data;
}

namespace sfem {
namespace codegen {

static const KernelDiagnostics vector_mass_tet4_jacobian_u0_u2_diagnostics_data = {
  "vector_mass_tet4_jacobian_u0_u2",
  "TET4",
  3,
  4,
  4,
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
  1,
  0,
  1012,
  1464,
  0,
  0,
  10,
  64,
  4,
  0,
  0,
  0,
  12,
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

extern "C" const sfem::codegen::KernelDiagnostics *cu_vector_mass_tet4_jacobian_u0_u2_diagnostics(void) {
  return &sfem::codegen::vector_mass_tet4_jacobian_u0_u2_diagnostics_data;
}

namespace sfem {
namespace codegen {

static const KernelDiagnostics vector_mass_tet4_jacobian_u1_u0_diagnostics_data = {
  "vector_mass_tet4_jacobian_u1_u0",
  "TET4",
  3,
  4,
  4,
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
  1,
  0,
  1012,
  1464,
  0,
  0,
  10,
  64,
  4,
  0,
  0,
  0,
  12,
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

extern "C" const sfem::codegen::KernelDiagnostics *cu_vector_mass_tet4_jacobian_u1_u0_diagnostics(void) {
  return &sfem::codegen::vector_mass_tet4_jacobian_u1_u0_diagnostics_data;
}

namespace sfem {
namespace codegen {

static const KernelDiagnostics vector_mass_tet4_jacobian_u1_u1_diagnostics_data = {
  "vector_mass_tet4_jacobian_u1_u1",
  "TET4",
  3,
  4,
  4,
  16,
  2,
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
  1012,
  1464,
  0,
  3,
  10,
  64,
  4,
  1,
  0,
  12,
  12,
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

extern "C" const sfem::codegen::KernelDiagnostics *cu_vector_mass_tet4_jacobian_u1_u1_diagnostics(void) {
  return &sfem::codegen::vector_mass_tet4_jacobian_u1_u1_diagnostics_data;
}

namespace sfem {
namespace codegen {

static const KernelDiagnostics vector_mass_tet4_jacobian_u1_u2_diagnostics_data = {
  "vector_mass_tet4_jacobian_u1_u2",
  "TET4",
  3,
  4,
  4,
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
  1,
  0,
  1012,
  1464,
  0,
  0,
  10,
  64,
  4,
  0,
  0,
  0,
  12,
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

extern "C" const sfem::codegen::KernelDiagnostics *cu_vector_mass_tet4_jacobian_u1_u2_diagnostics(void) {
  return &sfem::codegen::vector_mass_tet4_jacobian_u1_u2_diagnostics_data;
}

namespace sfem {
namespace codegen {

static const KernelDiagnostics vector_mass_tet4_jacobian_u2_u0_diagnostics_data = {
  "vector_mass_tet4_jacobian_u2_u0",
  "TET4",
  3,
  4,
  4,
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
  1,
  0,
  1012,
  1464,
  0,
  0,
  10,
  64,
  4,
  0,
  0,
  0,
  12,
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

extern "C" const sfem::codegen::KernelDiagnostics *cu_vector_mass_tet4_jacobian_u2_u0_diagnostics(void) {
  return &sfem::codegen::vector_mass_tet4_jacobian_u2_u0_diagnostics_data;
}

namespace sfem {
namespace codegen {

static const KernelDiagnostics vector_mass_tet4_jacobian_u2_u1_diagnostics_data = {
  "vector_mass_tet4_jacobian_u2_u1",
  "TET4",
  3,
  4,
  4,
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
  1,
  0,
  1012,
  1464,
  0,
  0,
  10,
  64,
  4,
  0,
  0,
  0,
  12,
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

extern "C" const sfem::codegen::KernelDiagnostics *cu_vector_mass_tet4_jacobian_u2_u1_diagnostics(void) {
  return &sfem::codegen::vector_mass_tet4_jacobian_u2_u1_diagnostics_data;
}

namespace sfem {
namespace codegen {

static const KernelDiagnostics vector_mass_tet4_jacobian_u2_u2_diagnostics_data = {
  "vector_mass_tet4_jacobian_u2_u2",
  "TET4",
  3,
  4,
  4,
  16,
  2,
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
  1012,
  1464,
  0,
  3,
  10,
  64,
  4,
  1,
  0,
  12,
  12,
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

extern "C" const sfem::codegen::KernelDiagnostics *cu_vector_mass_tet4_jacobian_u2_u2_diagnostics(void) {
  return &sfem::codegen::vector_mass_tet4_jacobian_u2_u2_diagnostics_data;
}

namespace sfem {
namespace codegen {

static const KernelDiagnostics vector_mass_tet4_jacobian_action_esoa_diagnostics_data = {
  "vector_mass_tet4_jacobian_action_esoa",
  "TET4",
  3,
  4,
  4,
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
  1012,
  1464,
  0,
  3,
  10,
  64,
  4,
  1,
  0,
  12,
  12,
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

extern "C" const sfem::codegen::KernelDiagnostics *cu_vector_mass_tet4_jacobian_action_esoa_diagnostics(void) {
  return &sfem::codegen::vector_mass_tet4_jacobian_action_esoa_diagnostics_data;
}

namespace sfem {
namespace codegen {

template <typename s_t, typename g_t>
__global__ void vector_mass_tet4_residual_a_msoa_impl(
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
  static constexpr int NQ = 4;
  static constexpr int NS = 4;
  static constexpr int NC = 3;
  const s_t *const affine_shape = sfem::codegen::ref_tet4_q4<s_t>::shape();
  const s_t *const affine_q_weight = sfem::codegen::quad_tet_q4<s_t>::q_weight();

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

    for (int stream = 0; stream < 12; ++stream) {
      {
        boutput[stream] = s_t(0);
      }
    }

    const g_t *const affine_geometry_sources[1] = {g_det0 + evb};
    s_t baffine_geometry_data[1];
    const s_t *bageom_streams[1];
    bageom_streams[0] = ageom_stream<s_t, g_t>(
        affine_geometry_sources[0], &baffine_geometry_data[0], std::is_same<g_t, s_t>());

    vector_mass_d3_simplex_tet4_residual_block_contiguous<s_t, NQ, NS>(0, bageom_streams[0], affine_shape, affine_q_weight, bcurrent, density, boutput);

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

extern "C" int cu_vector_mass_tet4_residual_a_msoa(
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
        if (nelements <= 0) return SFEM_SUCCESS;
        const int block_size = 256;
        const int grid_size = (int)((nelements + block_size - 1) / block_size);
        sfem::codegen::vector_mass_tet4_residual_a_msoa_impl<double, geom_t><<<grid_size, block_size, 0, (cudaStream_t)stream>>>(nelements, nnodes, elements, g_det0, density, current_stride, (const double *)u0, (const double *)u1, (const double *)u2, out_stride, (double *)u0_out, (double *)u1_out, (double *)u2_out);
        return sfem::codegen::launch_status("vector_mass_tet4_residual_a_msoa_impl");
    }
    case (int)sizeof(float): {
        if (nelements <= 0) return SFEM_SUCCESS;
        const int block_size = 256;
        const int grid_size = (int)((nelements + block_size - 1) / block_size);
        sfem::codegen::vector_mass_tet4_residual_a_msoa_impl<float, geom_t><<<grid_size, block_size, 0, (cudaStream_t)stream>>>(nelements, nnodes, elements, g_det0, density, current_stride, (const float *)u0, (const float *)u1, (const float *)u2, out_stride, (float *)u0_out, (float *)u1_out, (float *)u2_out);
        return sfem::codegen::launch_status("vector_mass_tet4_residual_a_msoa_impl");
    }
    default:
      break;
  }
  return sfem::codegen::unsupported_dispatch("vector_mass_tet4_residual_a_msoa", -1, (int)scalar_bytes);
}

namespace sfem {
namespace codegen {

template <typename s_t, typename g_t>
__global__ void vector_mass_tet4_jacobian_action_a_msoa_impl(
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
  static constexpr int NQ = 4;
  static constexpr int NS = 4;
  static constexpr int NC = 3;
  const s_t *const affine_shape = sfem::codegen::ref_tet4_q4<s_t>::shape();
  const s_t *const affine_q_weight = sfem::codegen::quad_tet_q4<s_t>::q_weight();

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

    for (int stream = 0; stream < 12; ++stream) {
      {
        boutput[stream] = s_t(0);
      }
    }

    const g_t *const affine_geometry_sources[1] = {g_det0 + evb};
    s_t baffine_geometry_data[1];
    const s_t *bageom_streams[1];
    bageom_streams[0] = ageom_stream<s_t, g_t>(
        affine_geometry_sources[0], &baffine_geometry_data[0], std::is_same<g_t, s_t>());

    vector_mass_d3_simplex_tet4_jacobian_action_block_contiguous<s_t, NQ, NS>(0, bageom_streams[0], affine_shape, affine_q_weight, bdirection, density, boutput);

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

extern "C" int cu_vector_mass_tet4_jacobian_action_a_msoa(
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
        if (nelements <= 0) return SFEM_SUCCESS;
        const int block_size = 256;
        const int grid_size = (int)((nelements + block_size - 1) / block_size);
        sfem::codegen::vector_mass_tet4_jacobian_action_a_msoa_impl<double, geom_t><<<grid_size, block_size, 0, (cudaStream_t)stream>>>(nelements, nnodes, elements, g_det0, density, direction_stride, (const double *)u0_direction, (const double *)u1_direction, (const double *)u2_direction, out_stride, (double *)u0_out, (double *)u1_out, (double *)u2_out);
        return sfem::codegen::launch_status("vector_mass_tet4_jacobian_action_a_msoa_impl");
    }
    case (int)sizeof(float): {
        if (nelements <= 0) return SFEM_SUCCESS;
        const int block_size = 256;
        const int grid_size = (int)((nelements + block_size - 1) / block_size);
        sfem::codegen::vector_mass_tet4_jacobian_action_a_msoa_impl<float, geom_t><<<grid_size, block_size, 0, (cudaStream_t)stream>>>(nelements, nnodes, elements, g_det0, density, direction_stride, (const float *)u0_direction, (const float *)u1_direction, (const float *)u2_direction, out_stride, (float *)u0_out, (float *)u1_out, (float *)u2_out);
        return sfem::codegen::launch_status("vector_mass_tet4_jacobian_action_a_msoa_impl");
    }
    default:
      break;
  }
  return sfem::codegen::unsupported_dispatch("vector_mass_tet4_jacobian_action_a_msoa", -1, (int)scalar_bytes);
}

namespace sfem {
namespace codegen {

__host__ __device__ __forceinline__ void vector_mass_tet4_hessian_crs_a_msoa_find_cols(
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

template <typename s_t, typename v_t>
__host__ __device__ __forceinline__ void vector_mass_tet4_hessian_crs_a_msoa_scatter_crs(
    const idx_t *const RSTR ev,
    const s_t *const RSTR element_matrix,
    const count_t *const RSTR rowptr,
    const idx_t *const RSTR colidx,
    v_t *const RSTR values) {
  static constexpr int NS = 4;
  static constexpr int NC = 3;
  static constexpr int N_COL_STREAMS = 12;
  count_t entries[NS * NS];
  idx_t ks[NS];
  for (int i = 0; i < NS; ++i) {
    const count_t row_begin = rowptr[ev[i]];
    const int lenrow = (int)(rowptr[ev[i] + 1] - row_begin);
    const idx_t *const RSTR cols = &colidx[row_begin];
    vector_mass_tet4_hessian_crs_a_msoa_find_cols(ev, cols, lenrow, ks);
    for (int j = 0; j < NS; ++j) {
      entries[i * NS + j] = row_begin + ks[j];
    }
  }
  for (int bi = 0; bi < NC; ++bi) {
    for (int row_shape = 0; row_shape < NS; ++row_shape) {
      const s_t *const RSTR row = &element_matrix[(bi * NS + row_shape) * N_COL_STREAMS];
      for (int bj = 0; bj < NC; ++bj) {
        for (int col_shape = 0; col_shape < NS; ++col_shape) {
          v_t *const block = &values[entries[row_shape * NS + col_shape] * NC * NC];
          atomicAdd(&(block[bi * NC + bj]), v_t(row[bj * NS + col_shape]));
        }
      }
    }
  }
}

template <typename s_t, typename v_t>
__global__ void vector_mass_tet4_hessian_crs_a_msoa_impl(
    const ptrdiff_t nelements,
    const ptrdiff_t,
    idx_t **const RSTR elements,
    const geom_t *const RSTR g_det0,
    const s_t density,
    const count_t *const RSTR rowptr,
    const idx_t *const RSTR colidx,
    v_t *const RSTR values
) {
  static constexpr int NQ = 4;
  static constexpr int NS = 4;
  const s_t *const isoparametric_shape = sfem::codegen::ref_tet4_q4<s_t>::shape();
  const s_t *const isoparametric_q_weight = sfem::codegen::quad_tet_q4<s_t>::q_weight();

  for (ptrdiff_t element = (ptrdiff_t)blockIdx.x * blockDim.x + threadIdx.x; element < nelements; element += (ptrdiff_t)blockDim.x * gridDim.x) {
    idx_t ev[NS];
    s_t element_matrix[144];
    s_t bdeterminant[NQ];

    for (int shape = 0; shape < NS; ++shape) {
      const idx_t node = elements[shape][element];
      ev[shape] = node;
    }


    bdeterminant[0] = s_t(g_det0[element]);

    vector_mass_d3_simplex_hessian_block<s_t, NQ, NS>(0, bdeterminant, isoparametric_shape, isoparametric_q_weight, density, element_matrix);

    vector_mass_tet4_hessian_crs_a_msoa_scatter_crs(ev, element_matrix, rowptr, colidx, values);
  }

}

} // namespace codegen
} // namespace sfem

extern "C" int cu_vector_mass_tet4_hessian_bsr_a_msoa(
    const int scalar_bytes,
    const int values_bytes,
    const ptrdiff_t nelements,
    const ptrdiff_t nnodes,
    idx_t **const RSTR elements,
    const geom_t *const RSTR g_det0,
    const real_t density,
    const count_t *const RSTR rowptr,
    const idx_t *const RSTR colidx,
    void *const RSTR values,
    void *const stream
) {
  switch (scalar_bytes) {
    case (int)sizeof(double): {
      switch (values_bytes) {
        case (int)sizeof(double): {
          if (nelements <= 0) return SFEM_SUCCESS;
          const int block_size = 256;
          const int grid_size = (int)((nelements + block_size - 1) / block_size);
          sfem::codegen::vector_mass_tet4_hessian_crs_a_msoa_impl<double, double><<<grid_size, block_size, 0, (cudaStream_t)stream>>>(nelements, nnodes, elements, g_det0, density, rowptr, colidx, (double *)values);
          return sfem::codegen::launch_status("vector_mass_tet4_hessian_crs_a_msoa_impl");
        }
        case (int)sizeof(float): {
          if (nelements <= 0) return SFEM_SUCCESS;
          const int block_size = 256;
          const int grid_size = (int)((nelements + block_size - 1) / block_size);
          sfem::codegen::vector_mass_tet4_hessian_crs_a_msoa_impl<double, float><<<grid_size, block_size, 0, (cudaStream_t)stream>>>(nelements, nnodes, elements, g_det0, density, rowptr, colidx, (float *)values);
          return sfem::codegen::launch_status("vector_mass_tet4_hessian_crs_a_msoa_impl");
        }
        default:
          break;
      }
      break;
    }
    case (int)sizeof(float): {
      switch (values_bytes) {
        case (int)sizeof(float): {
          if (nelements <= 0) return SFEM_SUCCESS;
          const int block_size = 256;
          const int grid_size = (int)((nelements + block_size - 1) / block_size);
          sfem::codegen::vector_mass_tet4_hessian_crs_a_msoa_impl<float, float><<<grid_size, block_size, 0, (cudaStream_t)stream>>>(nelements, nnodes, elements, g_det0, density, rowptr, colidx, (float *)values);
          return sfem::codegen::launch_status("vector_mass_tet4_hessian_crs_a_msoa_impl");
        }
        default:
          break;
      }
      break;
    }
    default:
      break;
  }
  return sfem::codegen::unsupported_values_dispatch("vector_mass_tet4_hessian_bsr_a_msoa", (int)scalar_bytes, (int)values_bytes);
}
