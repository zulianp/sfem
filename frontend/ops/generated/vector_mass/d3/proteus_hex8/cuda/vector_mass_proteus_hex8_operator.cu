#include <type_traits>
#include <cstdint>
#include <cstdlib>
#include <string.h>
#include "vector_mass_d3_tensor_product_local.cuh"
#include "geometry_kernels.cuh"
#include "kernel_diagnostics.cuh"
#include "line_p1_q2.cuh"
#include "quad_line_q2.cuh"
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

static const KernelDiagnostics vector_mass_proteus_hex8_residual_esoa_diagnostics_data = {
  "vector_mass_proteus_hex8_residual_esoa",
  "PROTEUS_HEX8",
  3,
  8,
  8,
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
  2456,
  3552,
  0,
  3,
  10,
  8,
  2,
  1,
  24,
  0,
  24,
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

extern "C" const sfem::codegen::KernelDiagnostics *cu_vector_mass_proteus_hex8_residual_esoa_diagnostics(void) {
  return &sfem::codegen::vector_mass_proteus_hex8_residual_esoa_diagnostics_data;
}

namespace sfem {
namespace codegen {

static const KernelDiagnostics vector_mass_proteus_hex8_jacobian_u0_u0_diagnostics_data = {
  "vector_mass_proteus_hex8_jacobian_u0_u0",
  "PROTEUS_HEX8",
  3,
  8,
  8,
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
  2456,
  3552,
  0,
  3,
  10,
  8,
  2,
  1,
  0,
  24,
  24,
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

extern "C" const sfem::codegen::KernelDiagnostics *cu_vector_mass_proteus_hex8_jacobian_u0_u0_diagnostics(void) {
  return &sfem::codegen::vector_mass_proteus_hex8_jacobian_u0_u0_diagnostics_data;
}

namespace sfem {
namespace codegen {

static const KernelDiagnostics vector_mass_proteus_hex8_jacobian_u0_u1_diagnostics_data = {
  "vector_mass_proteus_hex8_jacobian_u0_u1",
  "PROTEUS_HEX8",
  3,
  8,
  8,
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
  2456,
  3552,
  0,
  0,
  10,
  8,
  2,
  0,
  0,
  0,
  24,
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

extern "C" const sfem::codegen::KernelDiagnostics *cu_vector_mass_proteus_hex8_jacobian_u0_u1_diagnostics(void) {
  return &sfem::codegen::vector_mass_proteus_hex8_jacobian_u0_u1_diagnostics_data;
}

namespace sfem {
namespace codegen {

static const KernelDiagnostics vector_mass_proteus_hex8_jacobian_u0_u2_diagnostics_data = {
  "vector_mass_proteus_hex8_jacobian_u0_u2",
  "PROTEUS_HEX8",
  3,
  8,
  8,
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
  2456,
  3552,
  0,
  0,
  10,
  8,
  2,
  0,
  0,
  0,
  24,
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

extern "C" const sfem::codegen::KernelDiagnostics *cu_vector_mass_proteus_hex8_jacobian_u0_u2_diagnostics(void) {
  return &sfem::codegen::vector_mass_proteus_hex8_jacobian_u0_u2_diagnostics_data;
}

namespace sfem {
namespace codegen {

static const KernelDiagnostics vector_mass_proteus_hex8_jacobian_u1_u0_diagnostics_data = {
  "vector_mass_proteus_hex8_jacobian_u1_u0",
  "PROTEUS_HEX8",
  3,
  8,
  8,
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
  2456,
  3552,
  0,
  0,
  10,
  8,
  2,
  0,
  0,
  0,
  24,
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

extern "C" const sfem::codegen::KernelDiagnostics *cu_vector_mass_proteus_hex8_jacobian_u1_u0_diagnostics(void) {
  return &sfem::codegen::vector_mass_proteus_hex8_jacobian_u1_u0_diagnostics_data;
}

namespace sfem {
namespace codegen {

static const KernelDiagnostics vector_mass_proteus_hex8_jacobian_u1_u1_diagnostics_data = {
  "vector_mass_proteus_hex8_jacobian_u1_u1",
  "PROTEUS_HEX8",
  3,
  8,
  8,
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
  2456,
  3552,
  0,
  3,
  10,
  8,
  2,
  1,
  0,
  24,
  24,
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

extern "C" const sfem::codegen::KernelDiagnostics *cu_vector_mass_proteus_hex8_jacobian_u1_u1_diagnostics(void) {
  return &sfem::codegen::vector_mass_proteus_hex8_jacobian_u1_u1_diagnostics_data;
}

namespace sfem {
namespace codegen {

static const KernelDiagnostics vector_mass_proteus_hex8_jacobian_u1_u2_diagnostics_data = {
  "vector_mass_proteus_hex8_jacobian_u1_u2",
  "PROTEUS_HEX8",
  3,
  8,
  8,
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
  2456,
  3552,
  0,
  0,
  10,
  8,
  2,
  0,
  0,
  0,
  24,
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

extern "C" const sfem::codegen::KernelDiagnostics *cu_vector_mass_proteus_hex8_jacobian_u1_u2_diagnostics(void) {
  return &sfem::codegen::vector_mass_proteus_hex8_jacobian_u1_u2_diagnostics_data;
}

namespace sfem {
namespace codegen {

static const KernelDiagnostics vector_mass_proteus_hex8_jacobian_u2_u0_diagnostics_data = {
  "vector_mass_proteus_hex8_jacobian_u2_u0",
  "PROTEUS_HEX8",
  3,
  8,
  8,
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
  2456,
  3552,
  0,
  0,
  10,
  8,
  2,
  0,
  0,
  0,
  24,
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

extern "C" const sfem::codegen::KernelDiagnostics *cu_vector_mass_proteus_hex8_jacobian_u2_u0_diagnostics(void) {
  return &sfem::codegen::vector_mass_proteus_hex8_jacobian_u2_u0_diagnostics_data;
}

namespace sfem {
namespace codegen {

static const KernelDiagnostics vector_mass_proteus_hex8_jacobian_u2_u1_diagnostics_data = {
  "vector_mass_proteus_hex8_jacobian_u2_u1",
  "PROTEUS_HEX8",
  3,
  8,
  8,
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
  2456,
  3552,
  0,
  0,
  10,
  8,
  2,
  0,
  0,
  0,
  24,
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

extern "C" const sfem::codegen::KernelDiagnostics *cu_vector_mass_proteus_hex8_jacobian_u2_u1_diagnostics(void) {
  return &sfem::codegen::vector_mass_proteus_hex8_jacobian_u2_u1_diagnostics_data;
}

namespace sfem {
namespace codegen {

static const KernelDiagnostics vector_mass_proteus_hex8_jacobian_u2_u2_diagnostics_data = {
  "vector_mass_proteus_hex8_jacobian_u2_u2",
  "PROTEUS_HEX8",
  3,
  8,
  8,
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
  2456,
  3552,
  0,
  3,
  10,
  8,
  2,
  1,
  0,
  24,
  24,
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

extern "C" const sfem::codegen::KernelDiagnostics *cu_vector_mass_proteus_hex8_jacobian_u2_u2_diagnostics(void) {
  return &sfem::codegen::vector_mass_proteus_hex8_jacobian_u2_u2_diagnostics_data;
}

namespace sfem {
namespace codegen {

static const KernelDiagnostics vector_mass_proteus_hex8_jacobian_action_esoa_diagnostics_data = {
  "vector_mass_proteus_hex8_jacobian_action_esoa",
  "PROTEUS_HEX8",
  3,
  8,
  8,
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
  2456,
  3552,
  0,
  3,
  10,
  8,
  2,
  1,
  0,
  24,
  24,
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

extern "C" const sfem::codegen::KernelDiagnostics *cu_vector_mass_proteus_hex8_jacobian_action_esoa_diagnostics(void) {
  return &sfem::codegen::vector_mass_proteus_hex8_jacobian_action_esoa_diagnostics_data;
}

extern "C" int cu_vector_mass_proteus_hex8_residual_esoa(
    const int scalar_bytes,
    const ptrdiff_t geometry_stride,
    const void *const RSTR determinant,
    const void *const RSTR current[24],
    const real_t density,
    void *const RSTR output[24],
    void *const stream
) {
  switch (scalar_bytes) {
    case (int)sizeof(double): {
        sfem::codegen::vector_mass_d3_tensor_product_residual_block<double, 8, 8>(geometry_stride, (const double *)determinant, sfem::codegen::ref_line_p1_q2<double>::shape_1d(), sfem::codegen::quad_line_q2<double>::q_weight_1d(), (const double *const *)current, density, (double *const *)output);
        return SFEM_SUCCESS;
    }
    case (int)sizeof(float): {
        sfem::codegen::vector_mass_d3_tensor_product_residual_block<float, 8, 8>(geometry_stride, (const float *)determinant, sfem::codegen::ref_line_p1_q2<float>::shape_1d(), sfem::codegen::quad_line_q2<float>::q_weight_1d(), (const float *const *)current, density, (float *const *)output);
        return SFEM_SUCCESS;
    }
    default:
      break;
  }
  return sfem::codegen::unsupported_dispatch("vector_mass_proteus_hex8_residual_esoa", -1, (int)scalar_bytes);
}

namespace sfem {
namespace codegen {

template <typename s_t, typename g_t>
__global__ void vector_mass_proteus_hex8_residual_a_msoa_impl(
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
  static constexpr int NQ = 8;
  static constexpr int NS = 8;
  static constexpr int NC = 3;
  const s_t *const affine_shape_1d = sfem::codegen::ref_line_p1_q2<s_t>::shape_1d();
  const s_t *const affine_q_weight_1d = sfem::codegen::quad_line_q2<s_t>::q_weight_1d();

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

    for (int stream = 0; stream < 24; ++stream) {
      {
        boutput[stream] = s_t(0);
      }
    }

    const g_t *const affine_geometry_sources[1] = {g_det0 + evb};
    s_t baffine_geometry_data[1];
    const s_t *bageom_streams[1];
    bageom_streams[0] = ageom_stream<s_t, g_t>(
        affine_geometry_sources[0], &baffine_geometry_data[0], std::is_same<g_t, s_t>());

    vector_mass_d3_tensor_product_residual_block_contiguous<s_t, NQ, NS>(0, bageom_streams[0], affine_shape_1d, affine_q_weight_1d, bcurrent, density, boutput);

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

extern "C" int cu_vector_mass_proteus_hex8_residual_a_msoa(
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
        sfem::codegen::vector_mass_proteus_hex8_residual_a_msoa_impl<double, geom_t><<<grid_size, block_size, 0, (cudaStream_t)stream>>>(nelements, nnodes, elements, g_det0, density, current_stride, (const double *)u0, (const double *)u1, (const double *)u2, out_stride, (double *)u0_out, (double *)u1_out, (double *)u2_out);
        return sfem::codegen::launch_status("vector_mass_proteus_hex8_residual_a_msoa_impl");
    }
    case (int)sizeof(float): {
        const int block_size = 256;
        const int grid_size = (int)((nelements + block_size - 1) / block_size);
        sfem::codegen::vector_mass_proteus_hex8_residual_a_msoa_impl<float, geom_t><<<grid_size, block_size, 0, (cudaStream_t)stream>>>(nelements, nnodes, elements, g_det0, density, current_stride, (const float *)u0, (const float *)u1, (const float *)u2, out_stride, (float *)u0_out, (float *)u1_out, (float *)u2_out);
        return sfem::codegen::launch_status("vector_mass_proteus_hex8_residual_a_msoa_impl");
    }
    default:
      break;
  }
  return sfem::codegen::unsupported_dispatch("vector_mass_proteus_hex8_residual_a_msoa", -1, (int)scalar_bytes);
}

namespace sfem {
namespace codegen {

template <typename s_t>
__global__ void vector_mass_proteus_hex8_residual_i_msoa_impl(
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
  static constexpr int NQ = 8;
  static constexpr int NS = 8;
  static constexpr int NC = 3;
  const s_t *const isoparametric_shape_1d = sfem::codegen::ref_line_p1_q2<s_t>::shape_1d();
  const s_t *const isoparametric_grad_1d = sfem::codegen::ref_line_p1_q2<s_t>::grad_1d();
  const s_t *const isoparametric_q_weight_1d = sfem::codegen::quad_line_q2<s_t>::q_weight_1d();

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

    for (int stream = 0; stream < 24; ++stream) {
      {
        boutput[stream] = s_t(0);
      }
    }

    s_t coordinate_grad_ref[ND * NQ * ND];
    tensor_gradient_contiguous_scalar<s_t, NQ, NS, 3>(
        isoparametric_shape_1d, isoparametric_grad_1d, bcoordinates, 0,
        coordinate_grad_ref + 0);
    tensor_gradient_contiguous_scalar<s_t, NQ, NS, 3>(
        isoparametric_shape_1d, isoparametric_grad_1d, bcoordinates, 1,
        coordinate_grad_ref + NQ * ND);
    tensor_gradient_contiguous_scalar<s_t, NQ, NS, 3>(
        isoparametric_shape_1d, isoparametric_grad_1d, bcoordinates, 2,
        coordinate_grad_ref + 2 * NQ * ND);

    s_t *coordinate_grad_ref_adjugate_streams[ND * ND] = {badjugate_data[0], badjugate_data[1], badjugate_data[2], badjugate_data[3], badjugate_data[4], badjugate_data[5], badjugate_data[6], badjugate_data[7], badjugate_data[8]};
    geometry_jacobian_adjugate_and_determinant_scalar<s_t, ND, NQ>(
        coordinate_grad_ref, coordinate_grad_ref_adjugate_streams, bdeterminant);


    vector_mass_d3_tensor_product_residual_block_contiguous<s_t, NQ, NS>(1, bdeterminant, isoparametric_shape_1d, isoparametric_q_weight_1d, bcurrent, density, boutput);

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

extern "C" int cu_vector_mass_proteus_hex8_residual_i_msoa(
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
        sfem::codegen::vector_mass_proteus_hex8_residual_i_msoa_impl<double><<<grid_size, block_size, 0, (cudaStream_t)stream>>>(nelements, nnodes, elements, points, density, current_stride, (const double *)u0, (const double *)u1, (const double *)u2, out_stride, (double *)u0_out, (double *)u1_out, (double *)u2_out);
        return sfem::codegen::launch_status("vector_mass_proteus_hex8_residual_i_msoa_impl");
    }
    case (int)sizeof(float): {
        const int block_size = 256;
        const int grid_size = (int)((nelements + block_size - 1) / block_size);
        sfem::codegen::vector_mass_proteus_hex8_residual_i_msoa_impl<float><<<grid_size, block_size, 0, (cudaStream_t)stream>>>(nelements, nnodes, elements, points, density, current_stride, (const float *)u0, (const float *)u1, (const float *)u2, out_stride, (float *)u0_out, (float *)u1_out, (float *)u2_out);
        return sfem::codegen::launch_status("vector_mass_proteus_hex8_residual_i_msoa_impl");
    }
    default:
      break;
  }
  return sfem::codegen::unsupported_dispatch("vector_mass_proteus_hex8_residual_i_msoa", -1, (int)scalar_bytes);
}

extern "C" int cu_vector_mass_proteus_hex8_residual_i_maos(
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
        return cu_vector_mass_proteus_hex8_residual_i_msoa(scalar_bytes, nelements, nnodes, elements, points, ((const double *)parameters)[0], 3, (const double *)current + 0, (const double *)current + 1, (const double *)current + 2, 3, (double *)output + 0, (double *)output + 1, (double *)output + 2, stream);
    }
    case (int)sizeof(float): {
        return cu_vector_mass_proteus_hex8_residual_i_msoa(scalar_bytes, nelements, nnodes, elements, points, ((const float *)parameters)[0], 3, (const float *)current + 0, (const float *)current + 1, (const float *)current + 2, 3, (float *)output + 0, (float *)output + 1, (float *)output + 2, stream);
    }
    default:
      break;
  }
  return sfem::codegen::unsupported_dispatch("vector_mass_proteus_hex8_residual_i_maos", -1, (int)scalar_bytes);
}

extern "C" int cu_vector_mass_proteus_hex8_jacobian_action_esoa(
    const int scalar_bytes,
    const ptrdiff_t geometry_stride,
    const void *const RSTR determinant,
    const void *const RSTR direction[24],
    const real_t density,
    void *const RSTR output[24],
    void *const stream
) {
  switch (scalar_bytes) {
    case (int)sizeof(double): {
        sfem::codegen::vector_mass_d3_tensor_product_jacobian_action_block<double, 8, 8>(geometry_stride, (const double *)determinant, sfem::codegen::ref_line_p1_q2<double>::shape_1d(), sfem::codegen::quad_line_q2<double>::q_weight_1d(), (const double *const *)direction, density, (double *const *)output);
        return SFEM_SUCCESS;
    }
    case (int)sizeof(float): {
        sfem::codegen::vector_mass_d3_tensor_product_jacobian_action_block<float, 8, 8>(geometry_stride, (const float *)determinant, sfem::codegen::ref_line_p1_q2<float>::shape_1d(), sfem::codegen::quad_line_q2<float>::q_weight_1d(), (const float *const *)direction, density, (float *const *)output);
        return SFEM_SUCCESS;
    }
    default:
      break;
  }
  return sfem::codegen::unsupported_dispatch("vector_mass_proteus_hex8_jacobian_action_esoa", -1, (int)scalar_bytes);
}

namespace sfem {
namespace codegen {

template <typename s_t, typename g_t>
__global__ void vector_mass_proteus_hex8_jacobian_action_a_msoa_impl(
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
  static constexpr int NQ = 8;
  static constexpr int NS = 8;
  static constexpr int NC = 3;
  const s_t *const affine_shape_1d = sfem::codegen::ref_line_p1_q2<s_t>::shape_1d();
  const s_t *const affine_q_weight_1d = sfem::codegen::quad_line_q2<s_t>::q_weight_1d();

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

    for (int stream = 0; stream < 24; ++stream) {
      {
        boutput[stream] = s_t(0);
      }
    }

    const g_t *const affine_geometry_sources[1] = {g_det0 + evb};
    s_t baffine_geometry_data[1];
    const s_t *bageom_streams[1];
    bageom_streams[0] = ageom_stream<s_t, g_t>(
        affine_geometry_sources[0], &baffine_geometry_data[0], std::is_same<g_t, s_t>());

    vector_mass_d3_tensor_product_jacobian_action_block_contiguous<s_t, NQ, NS>(0, bageom_streams[0], affine_shape_1d, affine_q_weight_1d, bdirection, density, boutput);

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

extern "C" int cu_vector_mass_proteus_hex8_jacobian_action_a_msoa(
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
        sfem::codegen::vector_mass_proteus_hex8_jacobian_action_a_msoa_impl<double, geom_t><<<grid_size, block_size, 0, (cudaStream_t)stream>>>(nelements, nnodes, elements, g_det0, density, direction_stride, (const double *)u0_direction, (const double *)u1_direction, (const double *)u2_direction, out_stride, (double *)u0_out, (double *)u1_out, (double *)u2_out);
        return sfem::codegen::launch_status("vector_mass_proteus_hex8_jacobian_action_a_msoa_impl");
    }
    case (int)sizeof(float): {
        const int block_size = 256;
        const int grid_size = (int)((nelements + block_size - 1) / block_size);
        sfem::codegen::vector_mass_proteus_hex8_jacobian_action_a_msoa_impl<float, geom_t><<<grid_size, block_size, 0, (cudaStream_t)stream>>>(nelements, nnodes, elements, g_det0, density, direction_stride, (const float *)u0_direction, (const float *)u1_direction, (const float *)u2_direction, out_stride, (float *)u0_out, (float *)u1_out, (float *)u2_out);
        return sfem::codegen::launch_status("vector_mass_proteus_hex8_jacobian_action_a_msoa_impl");
    }
    default:
      break;
  }
  return sfem::codegen::unsupported_dispatch("vector_mass_proteus_hex8_jacobian_action_a_msoa", -1, (int)scalar_bytes);
}

namespace sfem {
namespace codegen {

template <typename s_t>
__global__ void vector_mass_proteus_hex8_jacobian_action_i_msoa_impl(
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
  static constexpr int NQ = 8;
  static constexpr int NS = 8;
  static constexpr int NC = 3;
  const s_t *const isoparametric_shape_1d = sfem::codegen::ref_line_p1_q2<s_t>::shape_1d();
  const s_t *const isoparametric_grad_1d = sfem::codegen::ref_line_p1_q2<s_t>::grad_1d();
  const s_t *const isoparametric_q_weight_1d = sfem::codegen::quad_line_q2<s_t>::q_weight_1d();

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

    for (int stream = 0; stream < 24; ++stream) {
      {
        boutput[stream] = s_t(0);
      }
    }

    s_t coordinate_grad_ref[ND * NQ * ND];
    tensor_gradient_contiguous_scalar<s_t, NQ, NS, 3>(
        isoparametric_shape_1d, isoparametric_grad_1d, bcoordinates, 0,
        coordinate_grad_ref + 0);
    tensor_gradient_contiguous_scalar<s_t, NQ, NS, 3>(
        isoparametric_shape_1d, isoparametric_grad_1d, bcoordinates, 1,
        coordinate_grad_ref + NQ * ND);
    tensor_gradient_contiguous_scalar<s_t, NQ, NS, 3>(
        isoparametric_shape_1d, isoparametric_grad_1d, bcoordinates, 2,
        coordinate_grad_ref + 2 * NQ * ND);

    s_t *coordinate_grad_ref_adjugate_streams[ND * ND] = {badjugate_data[0], badjugate_data[1], badjugate_data[2], badjugate_data[3], badjugate_data[4], badjugate_data[5], badjugate_data[6], badjugate_data[7], badjugate_data[8]};
    geometry_jacobian_adjugate_and_determinant_scalar<s_t, ND, NQ>(
        coordinate_grad_ref, coordinate_grad_ref_adjugate_streams, bdeterminant);


    vector_mass_d3_tensor_product_jacobian_action_block_contiguous<s_t, NQ, NS>(1, bdeterminant, isoparametric_shape_1d, isoparametric_q_weight_1d, bdirection, density, boutput);

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

extern "C" int cu_vector_mass_proteus_hex8_jacobian_action_i_msoa(
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
        sfem::codegen::vector_mass_proteus_hex8_jacobian_action_i_msoa_impl<double><<<grid_size, block_size, 0, (cudaStream_t)stream>>>(nelements, nnodes, elements, points, density, direction_stride, (const double *)u0_direction, (const double *)u1_direction, (const double *)u2_direction, out_stride, (double *)u0_out, (double *)u1_out, (double *)u2_out);
        return sfem::codegen::launch_status("vector_mass_proteus_hex8_jacobian_action_i_msoa_impl");
    }
    case (int)sizeof(float): {
        const int block_size = 256;
        const int grid_size = (int)((nelements + block_size - 1) / block_size);
        sfem::codegen::vector_mass_proteus_hex8_jacobian_action_i_msoa_impl<float><<<grid_size, block_size, 0, (cudaStream_t)stream>>>(nelements, nnodes, elements, points, density, direction_stride, (const float *)u0_direction, (const float *)u1_direction, (const float *)u2_direction, out_stride, (float *)u0_out, (float *)u1_out, (float *)u2_out);
        return sfem::codegen::launch_status("vector_mass_proteus_hex8_jacobian_action_i_msoa_impl");
    }
    default:
      break;
  }
  return sfem::codegen::unsupported_dispatch("vector_mass_proteus_hex8_jacobian_action_i_msoa", -1, (int)scalar_bytes);
}

extern "C" int cu_vector_mass_proteus_hex8_jacobian_action_i_maos(
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
        return cu_vector_mass_proteus_hex8_jacobian_action_i_msoa(scalar_bytes, nelements, nnodes, elements, points, ((const double *)parameters)[0], 3, (const double *)direction + 0, (const double *)direction + 1, (const double *)direction + 2, 3, (double *)output + 0, (double *)output + 1, (double *)output + 2, stream);
    }
    case (int)sizeof(float): {
        return cu_vector_mass_proteus_hex8_jacobian_action_i_msoa(scalar_bytes, nelements, nnodes, elements, points, ((const float *)parameters)[0], 3, (const float *)direction + 0, (const float *)direction + 1, (const float *)direction + 2, 3, (float *)output + 0, (float *)output + 1, (float *)output + 2, stream);
    }
    default:
      break;
  }
  return sfem::codegen::unsupported_dispatch("vector_mass_proteus_hex8_jacobian_action_i_maos", -1, (int)scalar_bytes);
}

namespace sfem {
namespace codegen {

__host__ __device__ __forceinline__ void vector_mass_proteus_hex8_hessian_crs_i_msoa_find_cols(
    const idx_t *const RSTR targets,
    const idx_t *const RSTR row,
    const int lenrow,
    idx_t *const RSTR ks) {
#pragma unroll(8)
  for (int d = 0; d < 8; ++d) {
    ks[d] = 0;
  }
  for (int k = 0; k < lenrow; ++k) {
#pragma unroll(8)
    for (int d = 0; d < 8; ++d) {
      ks[d] += row[k] < targets[d];
    }
  }
}

template <typename s_t>
__host__ __device__ __forceinline__ void vector_mass_proteus_hex8_hessian_crs_i_msoa_scatter_crs(
    const idx_t *const RSTR ev,
    const s_t *const RSTR element_matrix,
    const count_t *const RSTR rowptr,
    const idx_t *const RSTR colidx,
    s_t *const RSTR values) {
  static constexpr int NS = 8;
  static constexpr int NC = 3;
  static constexpr int N_COL_STREAMS = 24;
  count_t entries[NS * NS];
  idx_t ks[NS];
  for (int i = 0; i < NS; ++i) {
    const count_t row_begin = rowptr[ev[i]];
    const int lenrow = (int)(rowptr[ev[i] + 1] - row_begin);
    const idx_t *const RSTR cols = &colidx[row_begin];
    vector_mass_proteus_hex8_hessian_crs_i_msoa_find_cols(ev, cols, lenrow, ks);
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
__global__ void vector_mass_proteus_hex8_hessian_crs_i_msoa_impl(
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
  static constexpr int NQ = 8;
  static constexpr int NS = 8;
  const s_t *const isoparametric_shape_1d = sfem::codegen::ref_line_p1_q2<s_t>::shape_1d();
  const s_t *const isoparametric_grad_1d = sfem::codegen::ref_line_p1_q2<s_t>::grad_1d();
  const s_t *const isoparametric_q_weight_1d = sfem::codegen::quad_line_q2<s_t>::q_weight_1d();

  for (ptrdiff_t element = (ptrdiff_t)blockIdx.x * blockDim.x + threadIdx.x; element < nelements; element += (ptrdiff_t)blockDim.x * gridDim.x) {
    idx_t ev[NS];
    s_t element_matrix[576];
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

    s_t coordinate_grad_ref[ND * NQ * ND];
    tensor_gradient_contiguous_scalar<s_t, NQ, NS, 3>(
        isoparametric_shape_1d, isoparametric_grad_1d, bcoordinates, 0,
        coordinate_grad_ref + 0);
    tensor_gradient_contiguous_scalar<s_t, NQ, NS, 3>(
        isoparametric_shape_1d, isoparametric_grad_1d, bcoordinates, 1,
        coordinate_grad_ref + NQ * ND);
    tensor_gradient_contiguous_scalar<s_t, NQ, NS, 3>(
        isoparametric_shape_1d, isoparametric_grad_1d, bcoordinates, 2,
        coordinate_grad_ref + 2 * NQ * ND);

    s_t *coordinate_grad_ref_adjugate_streams[ND * ND] = {badjugate_data[0], badjugate_data[1], badjugate_data[2], badjugate_data[3], badjugate_data[4], badjugate_data[5], badjugate_data[6], badjugate_data[7], badjugate_data[8]};
    geometry_jacobian_adjugate_and_determinant_scalar<s_t, ND, NQ>(
        coordinate_grad_ref, coordinate_grad_ref_adjugate_streams, bdeterminant);

    vector_mass_d3_tensor_product_hessian_block<s_t, NQ, NS>(1, bdeterminant, isoparametric_shape_1d, isoparametric_q_weight_1d, density, element_matrix);

    vector_mass_proteus_hex8_hessian_crs_i_msoa_scatter_crs(ev, element_matrix, rowptr, colidx, values);
  }

}

} // namespace codegen
} // namespace sfem

extern "C" int cu_vector_mass_proteus_hex8_hessian_bsr_i_msoa(
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
      sfem::codegen::vector_mass_proteus_hex8_hessian_crs_i_msoa_impl<double><<<grid_size, block_size, 0, (cudaStream_t)stream>>>(nelements, nnodes, elements, points, density, rowptr, colidx, (double *)values);
      return sfem::codegen::launch_status("vector_mass_proteus_hex8_hessian_crs_i_msoa_impl");
    }
    case (int)sizeof(float): {
      const int block_size = 256;
      const int grid_size = (int)((nelements + block_size - 1) / block_size);
      sfem::codegen::vector_mass_proteus_hex8_hessian_crs_i_msoa_impl<float><<<grid_size, block_size, 0, (cudaStream_t)stream>>>(nelements, nnodes, elements, points, density, rowptr, colidx, (float *)values);
      return sfem::codegen::launch_status("vector_mass_proteus_hex8_hessian_crs_i_msoa_impl");
    }
    default:
      break;
  }
  return sfem::codegen::unsupported_dispatch("vector_mass_proteus_hex8_hessian_bsr_i_msoa", -1, (int)scalar_bytes);
}
