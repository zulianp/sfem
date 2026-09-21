#ifndef SAINT_VENANT_KIRCHHOFF_HEX8_ELEMENT_API_CUH
#define SAINT_VENANT_KIRCHHOFF_HEX8_ELEMENT_API_CUH

#include "../../proteus_hex8/cuda/saint_venant_kirchhoff_proteus_hex8_element.hpp"

namespace sfem {
namespace codegen {

template <typename s_t, int VS>
static __host__ __device__ __forceinline__ int saint_venant_kirchhoff_hex8_energy_egeometry_soa(
        const ptrdiff_t nelements,
        const s_t *const *const RSTR adj,
        const s_t *const RSTR det,
        const s_t lmbda,
        const s_t mu,
        const s_t *const *const RSTR u_streams,
        s_t *const RSTR values
) {
  static constexpr int NC = 3;
  static constexpr int NS = 8;
  static constexpr int NDOFS = NC * NS;
  const s_t *const ordered_u_streams[NDOFS] = {u_streams[0], u_streams[1], u_streams[2], u_streams[3], u_streams[4], u_streams[5], u_streams[9], u_streams[10], u_streams[11], u_streams[6], u_streams[7], u_streams[8], u_streams[12], u_streams[13], u_streams[14], u_streams[15], u_streams[16], u_streams[17], u_streams[21], u_streams[22], u_streams[23], u_streams[18], u_streams[19], u_streams[20]};
  return saint_venant_kirchhoff_proteus_hex8_energy_egeometry_soa<s_t, VS>(nelements, adj, det, lmbda, mu, ordered_u_streams, values);
}

template <typename s_t, int VS>
static __host__ __device__ __forceinline__ int saint_venant_kirchhoff_hex8_energy_ecoords_soa(
        const ptrdiff_t nelements,
        const s_t *const *const RSTR coords,
        const s_t lmbda,
        const s_t mu,
        const s_t *const *const RSTR u_streams,
        s_t *const RSTR values
) {
  static constexpr int NC = 3;
  static constexpr int NS = 8;
  static constexpr int NDOFS = NC * NS;
  const s_t *const ordered_coords[NDOFS] = {coords[0], coords[1], coords[2], coords[3], coords[4], coords[5], coords[9], coords[10], coords[11], coords[6], coords[7], coords[8], coords[12], coords[13], coords[14], coords[15], coords[16], coords[17], coords[21], coords[22], coords[23], coords[18], coords[19], coords[20]};
  const s_t *const ordered_u_streams[NDOFS] = {u_streams[0], u_streams[1], u_streams[2], u_streams[3], u_streams[4], u_streams[5], u_streams[9], u_streams[10], u_streams[11], u_streams[6], u_streams[7], u_streams[8], u_streams[12], u_streams[13], u_streams[14], u_streams[15], u_streams[16], u_streams[17], u_streams[21], u_streams[22], u_streams[23], u_streams[18], u_streams[19], u_streams[20]};
  return saint_venant_kirchhoff_proteus_hex8_energy_ecoords_soa<s_t, VS>(nelements, ordered_coords, lmbda, mu, ordered_u_streams, values);
}

template <typename s_t, int VS>
static __host__ __device__ __forceinline__ int saint_venant_kirchhoff_hex8_energy_esoa(
        const ptrdiff_t nelements,
        const s_t *const *const RSTR coords,
        const s_t lmbda,
        const s_t mu,
        const s_t *const *const RSTR u_streams,
        s_t *const RSTR values
) {
  static constexpr int NC = 3;
  static constexpr int NS = 8;
  static constexpr int NDOFS = NC * NS;
  const s_t *const ordered_coords[NDOFS] = {coords[0], coords[1], coords[2], coords[3], coords[4], coords[5], coords[9], coords[10], coords[11], coords[6], coords[7], coords[8], coords[12], coords[13], coords[14], coords[15], coords[16], coords[17], coords[21], coords[22], coords[23], coords[18], coords[19], coords[20]};
  const s_t *const ordered_u_streams[NDOFS] = {u_streams[0], u_streams[1], u_streams[2], u_streams[3], u_streams[4], u_streams[5], u_streams[9], u_streams[10], u_streams[11], u_streams[6], u_streams[7], u_streams[8], u_streams[12], u_streams[13], u_streams[14], u_streams[15], u_streams[16], u_streams[17], u_streams[21], u_streams[22], u_streams[23], u_streams[18], u_streams[19], u_streams[20]};
  return saint_venant_kirchhoff_proteus_hex8_energy_esoa<s_t, VS>(nelements, ordered_coords, lmbda, mu, ordered_u_streams, values);
}

template <typename s_t, int VS>
static __host__ __device__ __forceinline__ int saint_venant_kirchhoff_hex8_gradient_egeometry_soa(
        const ptrdiff_t nelements,
        const s_t *const *const RSTR adj,
        const s_t *const RSTR det,
        const s_t lmbda,
        const s_t mu,
        const s_t *const *const RSTR u_streams,
        s_t *const *const RSTR out_streams
) {
  static constexpr int NC = 3;
  static constexpr int NS = 8;
  static constexpr int NDOFS = NC * NS;
  const s_t *const ordered_u_streams[NDOFS] = {u_streams[0], u_streams[1], u_streams[2], u_streams[3], u_streams[4], u_streams[5], u_streams[9], u_streams[10], u_streams[11], u_streams[6], u_streams[7], u_streams[8], u_streams[12], u_streams[13], u_streams[14], u_streams[15], u_streams[16], u_streams[17], u_streams[21], u_streams[22], u_streams[23], u_streams[18], u_streams[19], u_streams[20]};
  s_t *const ordered_out_streams[NDOFS] = {out_streams[0], out_streams[1], out_streams[2], out_streams[3], out_streams[4], out_streams[5], out_streams[9], out_streams[10], out_streams[11], out_streams[6], out_streams[7], out_streams[8], out_streams[12], out_streams[13], out_streams[14], out_streams[15], out_streams[16], out_streams[17], out_streams[21], out_streams[22], out_streams[23], out_streams[18], out_streams[19], out_streams[20]};
  return saint_venant_kirchhoff_proteus_hex8_gradient_egeometry_soa<s_t, VS>(nelements, adj, det, lmbda, mu, ordered_u_streams, ordered_out_streams);
}

template <typename s_t, int VS>
static __host__ __device__ __forceinline__ int saint_venant_kirchhoff_hex8_gradient_ecoords_soa(
        const ptrdiff_t nelements,
        const s_t *const *const RSTR coords,
        const s_t lmbda,
        const s_t mu,
        const s_t *const *const RSTR u_streams,
        s_t *const *const RSTR out_streams
) {
  static constexpr int NC = 3;
  static constexpr int NS = 8;
  static constexpr int NDOFS = NC * NS;
  const s_t *const ordered_coords[NDOFS] = {coords[0], coords[1], coords[2], coords[3], coords[4], coords[5], coords[9], coords[10], coords[11], coords[6], coords[7], coords[8], coords[12], coords[13], coords[14], coords[15], coords[16], coords[17], coords[21], coords[22], coords[23], coords[18], coords[19], coords[20]};
  const s_t *const ordered_u_streams[NDOFS] = {u_streams[0], u_streams[1], u_streams[2], u_streams[3], u_streams[4], u_streams[5], u_streams[9], u_streams[10], u_streams[11], u_streams[6], u_streams[7], u_streams[8], u_streams[12], u_streams[13], u_streams[14], u_streams[15], u_streams[16], u_streams[17], u_streams[21], u_streams[22], u_streams[23], u_streams[18], u_streams[19], u_streams[20]};
  s_t *const ordered_out_streams[NDOFS] = {out_streams[0], out_streams[1], out_streams[2], out_streams[3], out_streams[4], out_streams[5], out_streams[9], out_streams[10], out_streams[11], out_streams[6], out_streams[7], out_streams[8], out_streams[12], out_streams[13], out_streams[14], out_streams[15], out_streams[16], out_streams[17], out_streams[21], out_streams[22], out_streams[23], out_streams[18], out_streams[19], out_streams[20]};
  return saint_venant_kirchhoff_proteus_hex8_gradient_ecoords_soa<s_t, VS>(nelements, ordered_coords, lmbda, mu, ordered_u_streams, ordered_out_streams);
}

template <typename s_t, int VS>
static __host__ __device__ __forceinline__ int saint_venant_kirchhoff_hex8_gradient_esoa(
        const ptrdiff_t nelements,
        const s_t *const *const RSTR coords,
        const s_t lmbda,
        const s_t mu,
        const s_t *const *const RSTR u_streams,
        s_t *const *const RSTR out_streams
) {
  static constexpr int NC = 3;
  static constexpr int NS = 8;
  static constexpr int NDOFS = NC * NS;
  const s_t *const ordered_coords[NDOFS] = {coords[0], coords[1], coords[2], coords[3], coords[4], coords[5], coords[9], coords[10], coords[11], coords[6], coords[7], coords[8], coords[12], coords[13], coords[14], coords[15], coords[16], coords[17], coords[21], coords[22], coords[23], coords[18], coords[19], coords[20]};
  const s_t *const ordered_u_streams[NDOFS] = {u_streams[0], u_streams[1], u_streams[2], u_streams[3], u_streams[4], u_streams[5], u_streams[9], u_streams[10], u_streams[11], u_streams[6], u_streams[7], u_streams[8], u_streams[12], u_streams[13], u_streams[14], u_streams[15], u_streams[16], u_streams[17], u_streams[21], u_streams[22], u_streams[23], u_streams[18], u_streams[19], u_streams[20]};
  s_t *const ordered_out_streams[NDOFS] = {out_streams[0], out_streams[1], out_streams[2], out_streams[3], out_streams[4], out_streams[5], out_streams[9], out_streams[10], out_streams[11], out_streams[6], out_streams[7], out_streams[8], out_streams[12], out_streams[13], out_streams[14], out_streams[15], out_streams[16], out_streams[17], out_streams[21], out_streams[22], out_streams[23], out_streams[18], out_streams[19], out_streams[20]};
  return saint_venant_kirchhoff_proteus_hex8_gradient_esoa<s_t, VS>(nelements, ordered_coords, lmbda, mu, ordered_u_streams, ordered_out_streams);
}

template <typename s_t, int VS>
static __host__ __device__ __forceinline__ int saint_venant_kirchhoff_hex8_hessian_egeometry_soa(
        const ptrdiff_t nelements,
        const s_t *const *const RSTR adj,
        const s_t *const RSTR det,
        const s_t lmbda,
        const s_t mu,
        const s_t *const *const RSTR u_streams,
        s_t *const *const RSTR matrix_streams
) {
  static constexpr int NC = 3;
  static constexpr int NS = 8;
  static constexpr int NDOFS = NC * NS;
  static constexpr int SHAPE_ORDER[NS] = {0, 1, 3, 2, 4, 5, 7, 6};
  const s_t *const ordered_u_streams[NDOFS] = {u_streams[0], u_streams[1], u_streams[2], u_streams[3], u_streams[4], u_streams[5], u_streams[9], u_streams[10], u_streams[11], u_streams[6], u_streams[7], u_streams[8], u_streams[12], u_streams[13], u_streams[14], u_streams[15], u_streams[16], u_streams[17], u_streams[21], u_streams[22], u_streams[23], u_streams[18], u_streams[19], u_streams[20]};
  s_t *ordered_matrix_streams[NDOFS * NDOFS];
  for (int row_shape = 0; row_shape < NS; ++row_shape) {
    const int source_row_shape = SHAPE_ORDER[row_shape];
    for (int row_component = 0; row_component < NC; ++row_component) {
      const int row = row_shape * NC + row_component;
      const int source_row = source_row_shape * NC + row_component;
      for (int col_shape = 0; col_shape < NS; ++col_shape) {
        const int source_col_shape = SHAPE_ORDER[col_shape];
        for (int col_component = 0; col_component < NC; ++col_component) {
          const int col = col_shape * NC + col_component;
          const int source_col = source_col_shape * NC + col_component;
          ordered_matrix_streams[row * NDOFS + col] = matrix_streams[source_row * NDOFS + source_col];
        }
      }
    }
  }
  return saint_venant_kirchhoff_proteus_hex8_hessian_egeometry_soa<s_t, VS>(nelements, adj, det, lmbda, mu, ordered_u_streams, ordered_matrix_streams);
}

template <typename s_t, int VS>
static __host__ __device__ __forceinline__ int saint_venant_kirchhoff_hex8_hessian_ecoords_soa(
        const ptrdiff_t nelements,
        const s_t *const *const RSTR coords,
        const s_t lmbda,
        const s_t mu,
        const s_t *const *const RSTR u_streams,
        s_t *const *const RSTR matrix_streams
) {
  static constexpr int NC = 3;
  static constexpr int NS = 8;
  static constexpr int NDOFS = NC * NS;
  static constexpr int SHAPE_ORDER[NS] = {0, 1, 3, 2, 4, 5, 7, 6};
  const s_t *const ordered_coords[NDOFS] = {coords[0], coords[1], coords[2], coords[3], coords[4], coords[5], coords[9], coords[10], coords[11], coords[6], coords[7], coords[8], coords[12], coords[13], coords[14], coords[15], coords[16], coords[17], coords[21], coords[22], coords[23], coords[18], coords[19], coords[20]};
  const s_t *const ordered_u_streams[NDOFS] = {u_streams[0], u_streams[1], u_streams[2], u_streams[3], u_streams[4], u_streams[5], u_streams[9], u_streams[10], u_streams[11], u_streams[6], u_streams[7], u_streams[8], u_streams[12], u_streams[13], u_streams[14], u_streams[15], u_streams[16], u_streams[17], u_streams[21], u_streams[22], u_streams[23], u_streams[18], u_streams[19], u_streams[20]};
  s_t *ordered_matrix_streams[NDOFS * NDOFS];
  for (int row_shape = 0; row_shape < NS; ++row_shape) {
    const int source_row_shape = SHAPE_ORDER[row_shape];
    for (int row_component = 0; row_component < NC; ++row_component) {
      const int row = row_shape * NC + row_component;
      const int source_row = source_row_shape * NC + row_component;
      for (int col_shape = 0; col_shape < NS; ++col_shape) {
        const int source_col_shape = SHAPE_ORDER[col_shape];
        for (int col_component = 0; col_component < NC; ++col_component) {
          const int col = col_shape * NC + col_component;
          const int source_col = source_col_shape * NC + col_component;
          ordered_matrix_streams[row * NDOFS + col] = matrix_streams[source_row * NDOFS + source_col];
        }
      }
    }
  }
  return saint_venant_kirchhoff_proteus_hex8_hessian_ecoords_soa<s_t, VS>(nelements, ordered_coords, lmbda, mu, ordered_u_streams, ordered_matrix_streams);
}

template <typename s_t, int VS>
static __host__ __device__ __forceinline__ int saint_venant_kirchhoff_hex8_hessian_esoa(
        const ptrdiff_t nelements,
        const s_t *const *const RSTR coords,
        const s_t lmbda,
        const s_t mu,
        const s_t *const *const RSTR u_streams,
        s_t *const *const RSTR matrix_streams
) {
  static constexpr int NC = 3;
  static constexpr int NS = 8;
  static constexpr int NDOFS = NC * NS;
  static constexpr int SHAPE_ORDER[NS] = {0, 1, 3, 2, 4, 5, 7, 6};
  const s_t *const ordered_coords[NDOFS] = {coords[0], coords[1], coords[2], coords[3], coords[4], coords[5], coords[9], coords[10], coords[11], coords[6], coords[7], coords[8], coords[12], coords[13], coords[14], coords[15], coords[16], coords[17], coords[21], coords[22], coords[23], coords[18], coords[19], coords[20]};
  const s_t *const ordered_u_streams[NDOFS] = {u_streams[0], u_streams[1], u_streams[2], u_streams[3], u_streams[4], u_streams[5], u_streams[9], u_streams[10], u_streams[11], u_streams[6], u_streams[7], u_streams[8], u_streams[12], u_streams[13], u_streams[14], u_streams[15], u_streams[16], u_streams[17], u_streams[21], u_streams[22], u_streams[23], u_streams[18], u_streams[19], u_streams[20]};
  s_t *ordered_matrix_streams[NDOFS * NDOFS];
  for (int row_shape = 0; row_shape < NS; ++row_shape) {
    const int source_row_shape = SHAPE_ORDER[row_shape];
    for (int row_component = 0; row_component < NC; ++row_component) {
      const int row = row_shape * NC + row_component;
      const int source_row = source_row_shape * NC + row_component;
      for (int col_shape = 0; col_shape < NS; ++col_shape) {
        const int source_col_shape = SHAPE_ORDER[col_shape];
        for (int col_component = 0; col_component < NC; ++col_component) {
          const int col = col_shape * NC + col_component;
          const int source_col = source_col_shape * NC + col_component;
          ordered_matrix_streams[row * NDOFS + col] = matrix_streams[source_row * NDOFS + source_col];
        }
      }
    }
  }
  return saint_venant_kirchhoff_proteus_hex8_hessian_esoa<s_t, VS>(nelements, ordered_coords, lmbda, mu, ordered_u_streams, ordered_matrix_streams);
}

} // namespace codegen
} // namespace sfem

#endif
