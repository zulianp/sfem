#include <type_traits>
#include <cuda_runtime.h>
#include "../../cuda/modified_mooney_rivlin_d3_tensor_product_local.cuh"
#include "../../../../reference/cuda/line_p2_q3.hpp"
#include "../../../../reference/cuda/quad_line_q3.hpp"
#include "../../../../reference/cuda/line_p2_q4.hpp"
#include "../../../../reference/cuda/quad_line_q4.hpp"
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

static const KernelDiagnostics modified_mooney_rivlin_hex27_objective_soa_diagnostics_data = {
  "modified_mooney_rivlin_hex27_objective_soa",
  "HEX27",
  3,
  64,
  27,
  16,
  4,
  32,
  31,
  0,
  0,
  19,
  0,
  1,
  0,
  6,
  9,
  102,
  11056,
  21024,
  8,
  17,
  10,
  24,
  4,
  2,
  81,
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

extern "C" const sfem::codegen::KernelDiagnostics *cu_modified_mooney_rivlin_hex27_objective_soa_diagnostics(void) {
  return &sfem::codegen::modified_mooney_rivlin_hex27_objective_soa_diagnostics_data;
}


namespace sfem {
namespace codegen {

template <typename s_t, typename g_t, int VS>
__global__ void modified_mooney_rivlin_hex27_objective_steps_a_msoa_impl(
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
        const s_t c1,
        const s_t c2,
        const s_t kappa,
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
  static constexpr int NQ = 27;
  static constexpr int NS = 27;
  const s_t *const affine_shape_1d = sfem::codegen::ref_line_p2_q3<s_t>::shape_1d();
  const s_t *const affine_grad_1d = sfem::codegen::ref_line_p2_q3<s_t>::grad_1d();
  const s_t *const affine_q_weight_1d = sfem::codegen::quad_line_q3<s_t>::q_weight_1d();

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
    const s_t *const bu_streams[NS * NC] = {bu_data[0], bu_data[1], bu_data[2], bu_data[24], bu_data[25], bu_data[26], bu_data[3], bu_data[4], bu_data[5], bu_data[33], bu_data[34], bu_data[35], bu_data[72], bu_data[73], bu_data[74], bu_data[27], bu_data[28], bu_data[29], bu_data[9], bu_data[10], bu_data[11], bu_data[30], bu_data[31], bu_data[32], bu_data[6], bu_data[7], bu_data[8], bu_data[48], bu_data[49], bu_data[50], bu_data[60], bu_data[61], bu_data[62], bu_data[51], bu_data[52], bu_data[53], bu_data[69], bu_data[70], bu_data[71], bu_data[78], bu_data[79], bu_data[80], bu_data[63], bu_data[64], bu_data[65], bu_data[57], bu_data[58], bu_data[59], bu_data[66], bu_data[67], bu_data[68], bu_data[54], bu_data[55], bu_data[56], bu_data[12], bu_data[13], bu_data[14], bu_data[36], bu_data[37], bu_data[38], bu_data[15], bu_data[16], bu_data[17], bu_data[45], bu_data[46], bu_data[47], bu_data[75], bu_data[76], bu_data[77], bu_data[39], bu_data[40], bu_data[41], bu_data[21], bu_data[22], bu_data[23], bu_data[42], bu_data[43], bu_data[44], bu_data[18], bu_data[19], bu_data[20]};
    const s_t *const bh_streams[NS * NC] = {bh_data[0], bh_data[1], bh_data[2], bh_data[24], bh_data[25], bh_data[26], bh_data[3], bh_data[4], bh_data[5], bh_data[33], bh_data[34], bh_data[35], bh_data[72], bh_data[73], bh_data[74], bh_data[27], bh_data[28], bh_data[29], bh_data[9], bh_data[10], bh_data[11], bh_data[30], bh_data[31], bh_data[32], bh_data[6], bh_data[7], bh_data[8], bh_data[48], bh_data[49], bh_data[50], bh_data[60], bh_data[61], bh_data[62], bh_data[51], bh_data[52], bh_data[53], bh_data[69], bh_data[70], bh_data[71], bh_data[78], bh_data[79], bh_data[80], bh_data[63], bh_data[64], bh_data[65], bh_data[57], bh_data[58], bh_data[59], bh_data[66], bh_data[67], bh_data[68], bh_data[54], bh_data[55], bh_data[56], bh_data[12], bh_data[13], bh_data[14], bh_data[36], bh_data[37], bh_data[38], bh_data[15], bh_data[16], bh_data[17], bh_data[45], bh_data[46], bh_data[47], bh_data[75], bh_data[76], bh_data[77], bh_data[39], bh_data[40], bh_data[41], bh_data[21], bh_data[22], bh_data[23], bh_data[42], bh_data[43], bh_data[44], bh_data[18], bh_data[19], bh_data[20]};

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

    modified_mooney_rivlin_d3_tensor_product_objective_block<s_t, NQ, NS, VS>(ne, 0, badj0, badj1, badj2, badj3, badj4, badj5, badj6, badj7, badj8, bdet0, affine_shape_1d, affine_grad_1d, affine_q_weight_1d, c1, c2, kappa, bu_streams, bh_streams, nsteps, steps, nelements, &value[evb]);
  }

}

} // namespace codegen
} // namespace sfem

extern "C" int cu_modified_mooney_rivlin_hex27_objective_steps_a_msoa(
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
        const real_t c1,
        const real_t c2,
        const real_t kappa,
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
        sfem::codegen::modified_mooney_rivlin_hex27_objective_steps_a_msoa_impl<double, geom_t, 1><<<grid_size, block_size, 0, (cudaStream_t)stream>>>(nelements, nnodes, elements, g_adj0, g_adj1, g_adj2, g_adj3, g_adj4, g_adj5, g_adj6, g_adj7, g_adj8, g_det0, c1, c2, kappa, u_stride, (const double *)ux, (const double *)uy, (const double *)uz, h_stride, (const double *)hx, (const double *)hy, (const double *)hz, nsteps, (const double *)steps, (double *)value);
        return sfem::codegen::launch_status("modified_mooney_rivlin_hex27_objective_steps_a_msoa_impl");
    }
    case (int)sizeof(float): {
        const int block_size = 256;
        const int grid_size = (int)((nelements + block_size - 1) / block_size);
        sfem::codegen::modified_mooney_rivlin_hex27_objective_steps_a_msoa_impl<float, geom_t, 1><<<grid_size, block_size, 0, (cudaStream_t)stream>>>(nelements, nnodes, elements, g_adj0, g_adj1, g_adj2, g_adj3, g_adj4, g_adj5, g_adj6, g_adj7, g_adj8, g_det0, c1, c2, kappa, u_stride, (const float *)ux, (const float *)uy, (const float *)uz, h_stride, (const float *)hx, (const float *)hy, (const float *)hz, nsteps, (const float *)steps, (float *)value);
        return sfem::codegen::launch_status("modified_mooney_rivlin_hex27_objective_steps_a_msoa_impl");
    }
    default:
      break;
  }
  return sfem::codegen::unsupported_dispatch("modified_mooney_rivlin_hex27_objective_steps_a_msoa", -1, (int)scalar_bytes);
}


namespace sfem {
namespace codegen {

template <typename s_t, typename g_t, int VS>
__global__ void modified_mooney_rivlin_hex27_objective_steps_i_msoa_impl(
        const ptrdiff_t nelements,
        const ptrdiff_t,
        idx_t **const RSTR elements,
        const g_t *const *const RSTR points,
        const s_t c1,
        const s_t c2,
        const s_t kappa,
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
  static constexpr int NQ = 64;
  static constexpr int NS = 27;
  const g_t *const RSTR x = points[0];
  const g_t *const RSTR y = points[1];
  const g_t *const RSTR z = points[2];
  const idx_t *const RSTR coordinate_elements[NS] = {elements[0], elements[8], elements[1], elements[11], elements[24], elements[9], elements[3], elements[10], elements[2], elements[16], elements[20], elements[17], elements[23], elements[26], elements[21], elements[19], elements[22], elements[18], elements[4], elements[12], elements[5], elements[15], elements[25], elements[13], elements[7], elements[14], elements[6]};
  const s_t *const isoparametric_shape_1d = sfem::codegen::ref_line_p2_q4<s_t>::shape_1d();
  const s_t *const isoparametric_grad_1d = sfem::codegen::ref_line_p2_q4<s_t>::grad_1d();
  const s_t *const isoparametric_q_weight_1d = sfem::codegen::quad_line_q4<s_t>::q_weight_1d();

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
      const idx_t *const RSTR coordinate_element_shape = coordinate_elements[shape];
      for (int d = 0; d < ND; ++d) {
        {
          bcoordinate_data[shape * ND + d][0] = coordinate_components[d][coordinate_element_shape[evb + 0]];
        }
      }
    }

    const s_t *const u_components[NC] = {ux, uy, uz};
    const s_t *const h_components[NC] = {hx, hy, hz};
    const s_t *const bu_streams[NS * NC] = {bu_data[0], bu_data[1], bu_data[2], bu_data[24], bu_data[25], bu_data[26], bu_data[3], bu_data[4], bu_data[5], bu_data[33], bu_data[34], bu_data[35], bu_data[72], bu_data[73], bu_data[74], bu_data[27], bu_data[28], bu_data[29], bu_data[9], bu_data[10], bu_data[11], bu_data[30], bu_data[31], bu_data[32], bu_data[6], bu_data[7], bu_data[8], bu_data[48], bu_data[49], bu_data[50], bu_data[60], bu_data[61], bu_data[62], bu_data[51], bu_data[52], bu_data[53], bu_data[69], bu_data[70], bu_data[71], bu_data[78], bu_data[79], bu_data[80], bu_data[63], bu_data[64], bu_data[65], bu_data[57], bu_data[58], bu_data[59], bu_data[66], bu_data[67], bu_data[68], bu_data[54], bu_data[55], bu_data[56], bu_data[12], bu_data[13], bu_data[14], bu_data[36], bu_data[37], bu_data[38], bu_data[15], bu_data[16], bu_data[17], bu_data[45], bu_data[46], bu_data[47], bu_data[75], bu_data[76], bu_data[77], bu_data[39], bu_data[40], bu_data[41], bu_data[21], bu_data[22], bu_data[23], bu_data[42], bu_data[43], bu_data[44], bu_data[18], bu_data[19], bu_data[20]};
    const s_t *const bh_streams[NS * NC] = {bh_data[0], bh_data[1], bh_data[2], bh_data[24], bh_data[25], bh_data[26], bh_data[3], bh_data[4], bh_data[5], bh_data[33], bh_data[34], bh_data[35], bh_data[72], bh_data[73], bh_data[74], bh_data[27], bh_data[28], bh_data[29], bh_data[9], bh_data[10], bh_data[11], bh_data[30], bh_data[31], bh_data[32], bh_data[6], bh_data[7], bh_data[8], bh_data[48], bh_data[49], bh_data[50], bh_data[60], bh_data[61], bh_data[62], bh_data[51], bh_data[52], bh_data[53], bh_data[69], bh_data[70], bh_data[71], bh_data[78], bh_data[79], bh_data[80], bh_data[63], bh_data[64], bh_data[65], bh_data[57], bh_data[58], bh_data[59], bh_data[66], bh_data[67], bh_data[68], bh_data[54], bh_data[55], bh_data[56], bh_data[12], bh_data[13], bh_data[14], bh_data[36], bh_data[37], bh_data[38], bh_data[15], bh_data[16], bh_data[17], bh_data[45], bh_data[46], bh_data[47], bh_data[75], bh_data[76], bh_data[77], bh_data[39], bh_data[40], bh_data[41], bh_data[21], bh_data[22], bh_data[23], bh_data[42], bh_data[43], bh_data[44], bh_data[18], bh_data[19], bh_data[20]};

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

    s_t coordinate_grad_ref[ND * NQ * ND * VS];
    tensor_gradient_contiguous<s_t, NQ, NS, VS, 3>(
        ne, isoparametric_shape_1d, isoparametric_grad_1d, bcoordinate_data, 0,
        coordinate_grad_ref + 0);
    tensor_gradient_contiguous<s_t, NQ, NS, VS, 3>(
        ne, isoparametric_shape_1d, isoparametric_grad_1d, bcoordinate_data, 1,
        coordinate_grad_ref + NQ * ND * VS);
    tensor_gradient_contiguous<s_t, NQ, NS, VS, 3>(
        ne, isoparametric_shape_1d, isoparametric_grad_1d, bcoordinate_data, 2,
        coordinate_grad_ref + 2 * NQ * ND * VS);

    s_t *coordinate_grad_ref_adjugate_streams[ND * ND] = {badj0, badj1, badj2, badj3, badj4, badj5, badj6, badj7, badj8};
    geometry_jacobian_adjugate_and_determinant<s_t, ND, NQ, VS>(
        ne, coordinate_grad_ref, coordinate_grad_ref_adjugate_streams, bdet0);

    for (int step = 0; step < nsteps; ++step) {
      {
        value[(ptrdiff_t)step * nelements + evb + 0] = s_t(0);
      }
    }

    modified_mooney_rivlin_d3_tensor_product_objective_block<s_t, NQ, NS, VS>(ne, VS, badj0, badj1, badj2, badj3, badj4, badj5, badj6, badj7, badj8, bdet0, isoparametric_shape_1d, isoparametric_grad_1d, isoparametric_q_weight_1d, c1, c2, kappa, bu_streams, bh_streams, nsteps, steps, nelements, &value[evb]);
  }

}

} // namespace codegen
} // namespace sfem

extern "C" int cu_modified_mooney_rivlin_hex27_objective_steps_i_msoa(
        const int scalar_bytes,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const real_t c1,
        const real_t c2,
        const real_t kappa,
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
        sfem::codegen::modified_mooney_rivlin_hex27_objective_steps_i_msoa_impl<double, geom_t, 1><<<grid_size, block_size, 0, (cudaStream_t)stream>>>(nelements, nnodes, elements, points, c1, c2, kappa, u_stride, (const double *)ux, (const double *)uy, (const double *)uz, h_stride, (const double *)hx, (const double *)hy, (const double *)hz, nsteps, (const double *)steps, (double *)value);
        return sfem::codegen::launch_status("modified_mooney_rivlin_hex27_objective_steps_i_msoa_impl");
    }
    case (int)sizeof(float): {
        const int block_size = 256;
        const int grid_size = (int)((nelements + block_size - 1) / block_size);
        sfem::codegen::modified_mooney_rivlin_hex27_objective_steps_i_msoa_impl<float, geom_t, 1><<<grid_size, block_size, 0, (cudaStream_t)stream>>>(nelements, nnodes, elements, points, c1, c2, kappa, u_stride, (const float *)ux, (const float *)uy, (const float *)uz, h_stride, (const float *)hx, (const float *)hy, (const float *)hz, nsteps, (const float *)steps, (float *)value);
        return sfem::codegen::launch_status("modified_mooney_rivlin_hex27_objective_steps_i_msoa_impl");
    }
    default:
      break;
  }
  return sfem::codegen::unsupported_dispatch("modified_mooney_rivlin_hex27_objective_steps_i_msoa", -1, (int)scalar_bytes);
}


namespace sfem {
namespace codegen {

static const KernelDiagnostics modified_mooney_rivlin_hex27_gradient_soa_diagnostics_data = {
  "modified_mooney_rivlin_hex27_gradient_soa",
  "HEX27",
  3,
  64,
  27,
  16,
  4,
  103,
  154,
  2,
  0,
  18,
  0,
  1,
  0,
  6,
  52,
  311,
  21988,
  31956,
  43,
  38,
  10,
  24,
  4,
  2,
  81,
  0,
  81,
  81,
  81,
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

extern "C" const sfem::codegen::KernelDiagnostics *cu_modified_mooney_rivlin_hex27_gradient_soa_diagnostics(void) {
  return &sfem::codegen::modified_mooney_rivlin_hex27_gradient_soa_diagnostics_data;
}


namespace sfem {
namespace codegen {

template <typename s_t, typename g_t, int VS>
__global__ void modified_mooney_rivlin_hex27_gradient_a_msoa_impl(
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
        const s_t c1,
        const s_t c2,
        const s_t kappa,
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
  static constexpr int NQ = 27;
  static constexpr int NS = 27;
  const s_t *const affine_shape_1d = sfem::codegen::ref_line_p2_q3<s_t>::shape_1d();
  const s_t *const affine_grad_1d = sfem::codegen::ref_line_p2_q3<s_t>::grad_1d();
  const s_t *const affine_q_weight_1d = sfem::codegen::quad_line_q3<s_t>::q_weight_1d();

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

    const s_t *const bu_streams[NS * NC] = {bu_data[0], bu_data[1], bu_data[2], bu_data[24], bu_data[25], bu_data[26], bu_data[3], bu_data[4], bu_data[5], bu_data[33], bu_data[34], bu_data[35], bu_data[72], bu_data[73], bu_data[74], bu_data[27], bu_data[28], bu_data[29], bu_data[9], bu_data[10], bu_data[11], bu_data[30], bu_data[31], bu_data[32], bu_data[6], bu_data[7], bu_data[8], bu_data[48], bu_data[49], bu_data[50], bu_data[60], bu_data[61], bu_data[62], bu_data[51], bu_data[52], bu_data[53], bu_data[69], bu_data[70], bu_data[71], bu_data[78], bu_data[79], bu_data[80], bu_data[63], bu_data[64], bu_data[65], bu_data[57], bu_data[58], bu_data[59], bu_data[66], bu_data[67], bu_data[68], bu_data[54], bu_data[55], bu_data[56], bu_data[12], bu_data[13], bu_data[14], bu_data[36], bu_data[37], bu_data[38], bu_data[15], bu_data[16], bu_data[17], bu_data[45], bu_data[46], bu_data[47], bu_data[75], bu_data[76], bu_data[77], bu_data[39], bu_data[40], bu_data[41], bu_data[21], bu_data[22], bu_data[23], bu_data[42], bu_data[43], bu_data[44], bu_data[18], bu_data[19], bu_data[20]};
    s_t *const bout_streams[NS * NC] = {bout_data[0], bout_data[1], bout_data[2], bout_data[24], bout_data[25], bout_data[26], bout_data[3], bout_data[4], bout_data[5], bout_data[33], bout_data[34], bout_data[35], bout_data[72], bout_data[73], bout_data[74], bout_data[27], bout_data[28], bout_data[29], bout_data[9], bout_data[10], bout_data[11], bout_data[30], bout_data[31], bout_data[32], bout_data[6], bout_data[7], bout_data[8], bout_data[48], bout_data[49], bout_data[50], bout_data[60], bout_data[61], bout_data[62], bout_data[51], bout_data[52], bout_data[53], bout_data[69], bout_data[70], bout_data[71], bout_data[78], bout_data[79], bout_data[80], bout_data[63], bout_data[64], bout_data[65], bout_data[57], bout_data[58], bout_data[59], bout_data[66], bout_data[67], bout_data[68], bout_data[54], bout_data[55], bout_data[56], bout_data[12], bout_data[13], bout_data[14], bout_data[36], bout_data[37], bout_data[38], bout_data[15], bout_data[16], bout_data[17], bout_data[45], bout_data[46], bout_data[47], bout_data[75], bout_data[76], bout_data[77], bout_data[39], bout_data[40], bout_data[41], bout_data[21], bout_data[22], bout_data[23], bout_data[42], bout_data[43], bout_data[44], bout_data[18], bout_data[19], bout_data[20]};
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

    modified_mooney_rivlin_d3_tensor_product_gradient_block<s_t, NQ, NS, VS>(ne, 0, badj0, badj1, badj2, badj3, badj4, badj5, badj6, badj7, badj8, bdet0, affine_shape_1d, affine_grad_1d, affine_q_weight_1d, c1, c2, kappa, bu_streams, bout_streams);

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

extern "C" int cu_modified_mooney_rivlin_hex27_gradient_a_msoa(
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
        const real_t c1,
        const real_t c2,
        const real_t kappa,
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
        sfem::codegen::modified_mooney_rivlin_hex27_gradient_a_msoa_impl<double, geom_t, 1><<<grid_size, block_size, 0, (cudaStream_t)stream>>>(nelements, nnodes, elements, g_adj0, g_adj1, g_adj2, g_adj3, g_adj4, g_adj5, g_adj6, g_adj7, g_adj8, g_det0, c1, c2, kappa, u_stride, (const double *)ux, (const double *)uy, (const double *)uz, out_stride, (double *)outx, (double *)outy, (double *)outz);
        return sfem::codegen::launch_status("modified_mooney_rivlin_hex27_gradient_a_msoa_impl");
    }
    case (int)sizeof(float): {
        const int block_size = 256;
        const int grid_size = (int)((nelements + block_size - 1) / block_size);
        sfem::codegen::modified_mooney_rivlin_hex27_gradient_a_msoa_impl<float, geom_t, 1><<<grid_size, block_size, 0, (cudaStream_t)stream>>>(nelements, nnodes, elements, g_adj0, g_adj1, g_adj2, g_adj3, g_adj4, g_adj5, g_adj6, g_adj7, g_adj8, g_det0, c1, c2, kappa, u_stride, (const float *)ux, (const float *)uy, (const float *)uz, out_stride, (float *)outx, (float *)outy, (float *)outz);
        return sfem::codegen::launch_status("modified_mooney_rivlin_hex27_gradient_a_msoa_impl");
    }
    default:
      break;
  }
  return sfem::codegen::unsupported_dispatch("modified_mooney_rivlin_hex27_gradient_a_msoa", -1, (int)scalar_bytes);
}


namespace sfem {
namespace codegen {

template <typename s_t, typename g_t, int VS>
__global__ void modified_mooney_rivlin_hex27_gradient_i_msoa_impl(
        const ptrdiff_t nelements,
        const ptrdiff_t,
        idx_t **const RSTR elements,
        const g_t *const *const RSTR points,
        const s_t c1,
        const s_t c2,
        const s_t kappa,
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
  static constexpr int NQ = 64;
  static constexpr int NS = 27;
  const g_t *const RSTR x = points[0];
  const g_t *const RSTR y = points[1];
  const g_t *const RSTR z = points[2];
  const idx_t *const RSTR coordinate_elements[NS] = {elements[0], elements[8], elements[1], elements[11], elements[24], elements[9], elements[3], elements[10], elements[2], elements[16], elements[20], elements[17], elements[23], elements[26], elements[21], elements[19], elements[22], elements[18], elements[4], elements[12], elements[5], elements[15], elements[25], elements[13], elements[7], elements[14], elements[6]};
  const s_t *const isoparametric_shape_1d = sfem::codegen::ref_line_p2_q4<s_t>::shape_1d();
  const s_t *const isoparametric_grad_1d = sfem::codegen::ref_line_p2_q4<s_t>::grad_1d();
  const s_t *const isoparametric_q_weight_1d = sfem::codegen::quad_line_q4<s_t>::q_weight_1d();

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
      const idx_t *const RSTR coordinate_element_shape = coordinate_elements[shape];
      for (int d = 0; d < ND; ++d) {
        {
          bcoordinate_data[shape * ND + d][0] = coordinate_components[d][coordinate_element_shape[evb + 0]];
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

    const s_t *const bu_streams[NS * NC] = {bu_data[0], bu_data[1], bu_data[2], bu_data[24], bu_data[25], bu_data[26], bu_data[3], bu_data[4], bu_data[5], bu_data[33], bu_data[34], bu_data[35], bu_data[72], bu_data[73], bu_data[74], bu_data[27], bu_data[28], bu_data[29], bu_data[9], bu_data[10], bu_data[11], bu_data[30], bu_data[31], bu_data[32], bu_data[6], bu_data[7], bu_data[8], bu_data[48], bu_data[49], bu_data[50], bu_data[60], bu_data[61], bu_data[62], bu_data[51], bu_data[52], bu_data[53], bu_data[69], bu_data[70], bu_data[71], bu_data[78], bu_data[79], bu_data[80], bu_data[63], bu_data[64], bu_data[65], bu_data[57], bu_data[58], bu_data[59], bu_data[66], bu_data[67], bu_data[68], bu_data[54], bu_data[55], bu_data[56], bu_data[12], bu_data[13], bu_data[14], bu_data[36], bu_data[37], bu_data[38], bu_data[15], bu_data[16], bu_data[17], bu_data[45], bu_data[46], bu_data[47], bu_data[75], bu_data[76], bu_data[77], bu_data[39], bu_data[40], bu_data[41], bu_data[21], bu_data[22], bu_data[23], bu_data[42], bu_data[43], bu_data[44], bu_data[18], bu_data[19], bu_data[20]};
    s_t *const bout_streams[NS * NC] = {bout_data[0], bout_data[1], bout_data[2], bout_data[24], bout_data[25], bout_data[26], bout_data[3], bout_data[4], bout_data[5], bout_data[33], bout_data[34], bout_data[35], bout_data[72], bout_data[73], bout_data[74], bout_data[27], bout_data[28], bout_data[29], bout_data[9], bout_data[10], bout_data[11], bout_data[30], bout_data[31], bout_data[32], bout_data[6], bout_data[7], bout_data[8], bout_data[48], bout_data[49], bout_data[50], bout_data[60], bout_data[61], bout_data[62], bout_data[51], bout_data[52], bout_data[53], bout_data[69], bout_data[70], bout_data[71], bout_data[78], bout_data[79], bout_data[80], bout_data[63], bout_data[64], bout_data[65], bout_data[57], bout_data[58], bout_data[59], bout_data[66], bout_data[67], bout_data[68], bout_data[54], bout_data[55], bout_data[56], bout_data[12], bout_data[13], bout_data[14], bout_data[36], bout_data[37], bout_data[38], bout_data[15], bout_data[16], bout_data[17], bout_data[45], bout_data[46], bout_data[47], bout_data[75], bout_data[76], bout_data[77], bout_data[39], bout_data[40], bout_data[41], bout_data[21], bout_data[22], bout_data[23], bout_data[42], bout_data[43], bout_data[44], bout_data[18], bout_data[19], bout_data[20]};

    s_t coordinate_grad_ref[ND * NQ * ND * VS];
    tensor_gradient_contiguous<s_t, NQ, NS, VS, 3>(
        ne, isoparametric_shape_1d, isoparametric_grad_1d, bcoordinate_data, 0,
        coordinate_grad_ref + 0);
    tensor_gradient_contiguous<s_t, NQ, NS, VS, 3>(
        ne, isoparametric_shape_1d, isoparametric_grad_1d, bcoordinate_data, 1,
        coordinate_grad_ref + NQ * ND * VS);
    tensor_gradient_contiguous<s_t, NQ, NS, VS, 3>(
        ne, isoparametric_shape_1d, isoparametric_grad_1d, bcoordinate_data, 2,
        coordinate_grad_ref + 2 * NQ * ND * VS);

    s_t *coordinate_grad_ref_adjugate_streams[ND * ND] = {badj0, badj1, badj2, badj3, badj4, badj5, badj6, badj7, badj8};
    geometry_jacobian_adjugate_and_determinant<s_t, ND, NQ, VS>(
        ne, coordinate_grad_ref, coordinate_grad_ref_adjugate_streams, bdet0);

    modified_mooney_rivlin_d3_tensor_product_gradient_block<s_t, NQ, NS, VS>(ne, VS, badj0, badj1, badj2, badj3, badj4, badj5, badj6, badj7, badj8, bdet0, isoparametric_shape_1d, isoparametric_grad_1d, isoparametric_q_weight_1d, c1, c2, kappa, bu_streams, bout_streams);

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

extern "C" int cu_modified_mooney_rivlin_hex27_gradient_i_msoa(
        const int scalar_bytes,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const real_t c1,
        const real_t c2,
        const real_t kappa,
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
        sfem::codegen::modified_mooney_rivlin_hex27_gradient_i_msoa_impl<double, geom_t, 1><<<grid_size, block_size, 0, (cudaStream_t)stream>>>(nelements, nnodes, elements, points, c1, c2, kappa, u_stride, (const double *)ux, (const double *)uy, (const double *)uz, out_stride, (double *)outx, (double *)outy, (double *)outz);
        return sfem::codegen::launch_status("modified_mooney_rivlin_hex27_gradient_i_msoa_impl");
    }
    case (int)sizeof(float): {
        const int block_size = 256;
        const int grid_size = (int)((nelements + block_size - 1) / block_size);
        sfem::codegen::modified_mooney_rivlin_hex27_gradient_i_msoa_impl<float, geom_t, 1><<<grid_size, block_size, 0, (cudaStream_t)stream>>>(nelements, nnodes, elements, points, c1, c2, kappa, u_stride, (const float *)ux, (const float *)uy, (const float *)uz, out_stride, (float *)outx, (float *)outy, (float *)outz);
        return sfem::codegen::launch_status("modified_mooney_rivlin_hex27_gradient_i_msoa_impl");
    }
    default:
      break;
  }
  return sfem::codegen::unsupported_dispatch("modified_mooney_rivlin_hex27_gradient_i_msoa", -1, (int)scalar_bytes);
}


namespace sfem {
namespace codegen {

static const KernelDiagnostics modified_mooney_rivlin_hex27_apply_soa_diagnostics_data = {
  "modified_mooney_rivlin_hex27_apply_soa",
  "HEX27",
  3,
  64,
  27,
  16,
  4,
  600,
  1076,
  2,
  0,
  30,
  0,
  1,
  0,
  6,
  331,
  1742,
  21988,
  31956,
  322,
  134,
  10,
  24,
  4,
  2,
  81,
  81,
  81,
  81,
  81,
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

extern "C" const sfem::codegen::KernelDiagnostics *cu_modified_mooney_rivlin_hex27_apply_soa_diagnostics(void) {
  return &sfem::codegen::modified_mooney_rivlin_hex27_apply_soa_diagnostics_data;
}


namespace sfem {
namespace codegen {

template <typename s_t, typename g_t, int VS>
__global__ void modified_mooney_rivlin_hex27_apply_a_msoa_impl(
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
        const s_t c1,
        const s_t c2,
        const s_t kappa,
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
  static constexpr int NQ = 27;
  static constexpr int NS = 27;
  const s_t *const affine_shape_1d = sfem::codegen::ref_line_p2_q3<s_t>::shape_1d();
  const s_t *const affine_grad_1d = sfem::codegen::ref_line_p2_q3<s_t>::grad_1d();
  const s_t *const affine_q_weight_1d = sfem::codegen::quad_line_q3<s_t>::q_weight_1d();

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

    const s_t *const bu_streams[NS * NC] = {bu_data[0], bu_data[1], bu_data[2], bu_data[24], bu_data[25], bu_data[26], bu_data[3], bu_data[4], bu_data[5], bu_data[33], bu_data[34], bu_data[35], bu_data[72], bu_data[73], bu_data[74], bu_data[27], bu_data[28], bu_data[29], bu_data[9], bu_data[10], bu_data[11], bu_data[30], bu_data[31], bu_data[32], bu_data[6], bu_data[7], bu_data[8], bu_data[48], bu_data[49], bu_data[50], bu_data[60], bu_data[61], bu_data[62], bu_data[51], bu_data[52], bu_data[53], bu_data[69], bu_data[70], bu_data[71], bu_data[78], bu_data[79], bu_data[80], bu_data[63], bu_data[64], bu_data[65], bu_data[57], bu_data[58], bu_data[59], bu_data[66], bu_data[67], bu_data[68], bu_data[54], bu_data[55], bu_data[56], bu_data[12], bu_data[13], bu_data[14], bu_data[36], bu_data[37], bu_data[38], bu_data[15], bu_data[16], bu_data[17], bu_data[45], bu_data[46], bu_data[47], bu_data[75], bu_data[76], bu_data[77], bu_data[39], bu_data[40], bu_data[41], bu_data[21], bu_data[22], bu_data[23], bu_data[42], bu_data[43], bu_data[44], bu_data[18], bu_data[19], bu_data[20]};
    const s_t *const bh_streams[NS * NC] = {bh_data[0], bh_data[1], bh_data[2], bh_data[24], bh_data[25], bh_data[26], bh_data[3], bh_data[4], bh_data[5], bh_data[33], bh_data[34], bh_data[35], bh_data[72], bh_data[73], bh_data[74], bh_data[27], bh_data[28], bh_data[29], bh_data[9], bh_data[10], bh_data[11], bh_data[30], bh_data[31], bh_data[32], bh_data[6], bh_data[7], bh_data[8], bh_data[48], bh_data[49], bh_data[50], bh_data[60], bh_data[61], bh_data[62], bh_data[51], bh_data[52], bh_data[53], bh_data[69], bh_data[70], bh_data[71], bh_data[78], bh_data[79], bh_data[80], bh_data[63], bh_data[64], bh_data[65], bh_data[57], bh_data[58], bh_data[59], bh_data[66], bh_data[67], bh_data[68], bh_data[54], bh_data[55], bh_data[56], bh_data[12], bh_data[13], bh_data[14], bh_data[36], bh_data[37], bh_data[38], bh_data[15], bh_data[16], bh_data[17], bh_data[45], bh_data[46], bh_data[47], bh_data[75], bh_data[76], bh_data[77], bh_data[39], bh_data[40], bh_data[41], bh_data[21], bh_data[22], bh_data[23], bh_data[42], bh_data[43], bh_data[44], bh_data[18], bh_data[19], bh_data[20]};
    s_t *const bout_streams[NS * NC] = {bout_data[0], bout_data[1], bout_data[2], bout_data[24], bout_data[25], bout_data[26], bout_data[3], bout_data[4], bout_data[5], bout_data[33], bout_data[34], bout_data[35], bout_data[72], bout_data[73], bout_data[74], bout_data[27], bout_data[28], bout_data[29], bout_data[9], bout_data[10], bout_data[11], bout_data[30], bout_data[31], bout_data[32], bout_data[6], bout_data[7], bout_data[8], bout_data[48], bout_data[49], bout_data[50], bout_data[60], bout_data[61], bout_data[62], bout_data[51], bout_data[52], bout_data[53], bout_data[69], bout_data[70], bout_data[71], bout_data[78], bout_data[79], bout_data[80], bout_data[63], bout_data[64], bout_data[65], bout_data[57], bout_data[58], bout_data[59], bout_data[66], bout_data[67], bout_data[68], bout_data[54], bout_data[55], bout_data[56], bout_data[12], bout_data[13], bout_data[14], bout_data[36], bout_data[37], bout_data[38], bout_data[15], bout_data[16], bout_data[17], bout_data[45], bout_data[46], bout_data[47], bout_data[75], bout_data[76], bout_data[77], bout_data[39], bout_data[40], bout_data[41], bout_data[21], bout_data[22], bout_data[23], bout_data[42], bout_data[43], bout_data[44], bout_data[18], bout_data[19], bout_data[20]};
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

    modified_mooney_rivlin_d3_tensor_product_apply_block<s_t, NQ, NS, VS>(ne, 0, badj0, badj1, badj2, badj3, badj4, badj5, badj6, badj7, badj8, bdet0, affine_shape_1d, affine_grad_1d, affine_q_weight_1d, c1, c2, kappa, bu_streams, bh_streams, bout_streams);

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

extern "C" int cu_modified_mooney_rivlin_hex27_apply_a_msoa(
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
        const real_t c1,
        const real_t c2,
        const real_t kappa,
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
        sfem::codegen::modified_mooney_rivlin_hex27_apply_a_msoa_impl<double, geom_t, 1><<<grid_size, block_size, 0, (cudaStream_t)stream>>>(nelements, nnodes, elements, g_adj0, g_adj1, g_adj2, g_adj3, g_adj4, g_adj5, g_adj6, g_adj7, g_adj8, g_det0, c1, c2, kappa, u_stride, (const double *)ux, (const double *)uy, (const double *)uz, h_stride, (const double *)hx, (const double *)hy, (const double *)hz, out_stride, (double *)outx, (double *)outy, (double *)outz);
        return sfem::codegen::launch_status("modified_mooney_rivlin_hex27_apply_a_msoa_impl");
    }
    case (int)sizeof(float): {
        const int block_size = 256;
        const int grid_size = (int)((nelements + block_size - 1) / block_size);
        sfem::codegen::modified_mooney_rivlin_hex27_apply_a_msoa_impl<float, geom_t, 1><<<grid_size, block_size, 0, (cudaStream_t)stream>>>(nelements, nnodes, elements, g_adj0, g_adj1, g_adj2, g_adj3, g_adj4, g_adj5, g_adj6, g_adj7, g_adj8, g_det0, c1, c2, kappa, u_stride, (const float *)ux, (const float *)uy, (const float *)uz, h_stride, (const float *)hx, (const float *)hy, (const float *)hz, out_stride, (float *)outx, (float *)outy, (float *)outz);
        return sfem::codegen::launch_status("modified_mooney_rivlin_hex27_apply_a_msoa_impl");
    }
    default:
      break;
  }
  return sfem::codegen::unsupported_dispatch("modified_mooney_rivlin_hex27_apply_a_msoa", -1, (int)scalar_bytes);
}


namespace sfem {
namespace codegen {

template <typename s_t, typename g_t, int VS>
__global__ void modified_mooney_rivlin_hex27_apply_i_msoa_impl(
        const ptrdiff_t nelements,
        const ptrdiff_t,
        idx_t **const RSTR elements,
        const g_t *const *const RSTR points,
        const s_t c1,
        const s_t c2,
        const s_t kappa,
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
  static constexpr int NQ = 64;
  static constexpr int NS = 27;
  const g_t *const RSTR x = points[0];
  const g_t *const RSTR y = points[1];
  const g_t *const RSTR z = points[2];
  const idx_t *const RSTR coordinate_elements[NS] = {elements[0], elements[8], elements[1], elements[11], elements[24], elements[9], elements[3], elements[10], elements[2], elements[16], elements[20], elements[17], elements[23], elements[26], elements[21], elements[19], elements[22], elements[18], elements[4], elements[12], elements[5], elements[15], elements[25], elements[13], elements[7], elements[14], elements[6]};
  const s_t *const isoparametric_shape_1d = sfem::codegen::ref_line_p2_q4<s_t>::shape_1d();
  const s_t *const isoparametric_grad_1d = sfem::codegen::ref_line_p2_q4<s_t>::grad_1d();
  const s_t *const isoparametric_q_weight_1d = sfem::codegen::quad_line_q4<s_t>::q_weight_1d();

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
      const idx_t *const RSTR coordinate_element_shape = coordinate_elements[shape];
      for (int d = 0; d < ND; ++d) {
        {
          bcoordinate_data[shape * ND + d][0] = coordinate_components[d][coordinate_element_shape[evb + 0]];
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

    const s_t *const bu_streams[NS * NC] = {bu_data[0], bu_data[1], bu_data[2], bu_data[24], bu_data[25], bu_data[26], bu_data[3], bu_data[4], bu_data[5], bu_data[33], bu_data[34], bu_data[35], bu_data[72], bu_data[73], bu_data[74], bu_data[27], bu_data[28], bu_data[29], bu_data[9], bu_data[10], bu_data[11], bu_data[30], bu_data[31], bu_data[32], bu_data[6], bu_data[7], bu_data[8], bu_data[48], bu_data[49], bu_data[50], bu_data[60], bu_data[61], bu_data[62], bu_data[51], bu_data[52], bu_data[53], bu_data[69], bu_data[70], bu_data[71], bu_data[78], bu_data[79], bu_data[80], bu_data[63], bu_data[64], bu_data[65], bu_data[57], bu_data[58], bu_data[59], bu_data[66], bu_data[67], bu_data[68], bu_data[54], bu_data[55], bu_data[56], bu_data[12], bu_data[13], bu_data[14], bu_data[36], bu_data[37], bu_data[38], bu_data[15], bu_data[16], bu_data[17], bu_data[45], bu_data[46], bu_data[47], bu_data[75], bu_data[76], bu_data[77], bu_data[39], bu_data[40], bu_data[41], bu_data[21], bu_data[22], bu_data[23], bu_data[42], bu_data[43], bu_data[44], bu_data[18], bu_data[19], bu_data[20]};
    const s_t *const bh_streams[NS * NC] = {bh_data[0], bh_data[1], bh_data[2], bh_data[24], bh_data[25], bh_data[26], bh_data[3], bh_data[4], bh_data[5], bh_data[33], bh_data[34], bh_data[35], bh_data[72], bh_data[73], bh_data[74], bh_data[27], bh_data[28], bh_data[29], bh_data[9], bh_data[10], bh_data[11], bh_data[30], bh_data[31], bh_data[32], bh_data[6], bh_data[7], bh_data[8], bh_data[48], bh_data[49], bh_data[50], bh_data[60], bh_data[61], bh_data[62], bh_data[51], bh_data[52], bh_data[53], bh_data[69], bh_data[70], bh_data[71], bh_data[78], bh_data[79], bh_data[80], bh_data[63], bh_data[64], bh_data[65], bh_data[57], bh_data[58], bh_data[59], bh_data[66], bh_data[67], bh_data[68], bh_data[54], bh_data[55], bh_data[56], bh_data[12], bh_data[13], bh_data[14], bh_data[36], bh_data[37], bh_data[38], bh_data[15], bh_data[16], bh_data[17], bh_data[45], bh_data[46], bh_data[47], bh_data[75], bh_data[76], bh_data[77], bh_data[39], bh_data[40], bh_data[41], bh_data[21], bh_data[22], bh_data[23], bh_data[42], bh_data[43], bh_data[44], bh_data[18], bh_data[19], bh_data[20]};
    s_t *const bout_streams[NS * NC] = {bout_data[0], bout_data[1], bout_data[2], bout_data[24], bout_data[25], bout_data[26], bout_data[3], bout_data[4], bout_data[5], bout_data[33], bout_data[34], bout_data[35], bout_data[72], bout_data[73], bout_data[74], bout_data[27], bout_data[28], bout_data[29], bout_data[9], bout_data[10], bout_data[11], bout_data[30], bout_data[31], bout_data[32], bout_data[6], bout_data[7], bout_data[8], bout_data[48], bout_data[49], bout_data[50], bout_data[60], bout_data[61], bout_data[62], bout_data[51], bout_data[52], bout_data[53], bout_data[69], bout_data[70], bout_data[71], bout_data[78], bout_data[79], bout_data[80], bout_data[63], bout_data[64], bout_data[65], bout_data[57], bout_data[58], bout_data[59], bout_data[66], bout_data[67], bout_data[68], bout_data[54], bout_data[55], bout_data[56], bout_data[12], bout_data[13], bout_data[14], bout_data[36], bout_data[37], bout_data[38], bout_data[15], bout_data[16], bout_data[17], bout_data[45], bout_data[46], bout_data[47], bout_data[75], bout_data[76], bout_data[77], bout_data[39], bout_data[40], bout_data[41], bout_data[21], bout_data[22], bout_data[23], bout_data[42], bout_data[43], bout_data[44], bout_data[18], bout_data[19], bout_data[20]};

    s_t coordinate_grad_ref[ND * NQ * ND * VS];
    tensor_gradient_contiguous<s_t, NQ, NS, VS, 3>(
        ne, isoparametric_shape_1d, isoparametric_grad_1d, bcoordinate_data, 0,
        coordinate_grad_ref + 0);
    tensor_gradient_contiguous<s_t, NQ, NS, VS, 3>(
        ne, isoparametric_shape_1d, isoparametric_grad_1d, bcoordinate_data, 1,
        coordinate_grad_ref + NQ * ND * VS);
    tensor_gradient_contiguous<s_t, NQ, NS, VS, 3>(
        ne, isoparametric_shape_1d, isoparametric_grad_1d, bcoordinate_data, 2,
        coordinate_grad_ref + 2 * NQ * ND * VS);

    s_t *coordinate_grad_ref_adjugate_streams[ND * ND] = {badj0, badj1, badj2, badj3, badj4, badj5, badj6, badj7, badj8};
    geometry_jacobian_adjugate_and_determinant<s_t, ND, NQ, VS>(
        ne, coordinate_grad_ref, coordinate_grad_ref_adjugate_streams, bdet0);

    modified_mooney_rivlin_d3_tensor_product_apply_block<s_t, NQ, NS, VS>(ne, VS, badj0, badj1, badj2, badj3, badj4, badj5, badj6, badj7, badj8, bdet0, isoparametric_shape_1d, isoparametric_grad_1d, isoparametric_q_weight_1d, c1, c2, kappa, bu_streams, bh_streams, bout_streams);

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

extern "C" int cu_modified_mooney_rivlin_hex27_apply_i_msoa(
        const int scalar_bytes,
        const ptrdiff_t nelements,
        const ptrdiff_t nnodes,
        idx_t **const RSTR elements,
        const geom_t *const *const RSTR points,
        const real_t c1,
        const real_t c2,
        const real_t kappa,
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
        sfem::codegen::modified_mooney_rivlin_hex27_apply_i_msoa_impl<double, geom_t, 1><<<grid_size, block_size, 0, (cudaStream_t)stream>>>(nelements, nnodes, elements, points, c1, c2, kappa, u_stride, (const double *)ux, (const double *)uy, (const double *)uz, h_stride, (const double *)hx, (const double *)hy, (const double *)hz, out_stride, (double *)outx, (double *)outy, (double *)outz);
        return sfem::codegen::launch_status("modified_mooney_rivlin_hex27_apply_i_msoa_impl");
    }
    case (int)sizeof(float): {
        const int block_size = 256;
        const int grid_size = (int)((nelements + block_size - 1) / block_size);
        sfem::codegen::modified_mooney_rivlin_hex27_apply_i_msoa_impl<float, geom_t, 1><<<grid_size, block_size, 0, (cudaStream_t)stream>>>(nelements, nnodes, elements, points, c1, c2, kappa, u_stride, (const float *)ux, (const float *)uy, (const float *)uz, h_stride, (const float *)hx, (const float *)hy, (const float *)hz, out_stride, (float *)outx, (float *)outy, (float *)outz);
        return sfem::codegen::launch_status("modified_mooney_rivlin_hex27_apply_i_msoa_impl");
    }
    default:
      break;
  }
  return sfem::codegen::unsupported_dispatch("modified_mooney_rivlin_hex27_apply_i_msoa", -1, (int)scalar_bytes);
}
