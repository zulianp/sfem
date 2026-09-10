#ifndef LAPLACE_HEX8_ELEMENT_API_HPP
#define LAPLACE_HEX8_ELEMENT_API_HPP

#include "../proteus_hex8/laplace_proteus_hex8_element.hpp"

namespace sfem {
namespace codegen {

template <typename s_t, int VS>
static SFEM_INLINE int laplace_hex8_energy_egeometry_soa(
        const ptrdiff_t nelements,
        const s_t *const *const RSTR adj,
        const s_t *const RSTR det,
        const s_t kappa,
        const s_t *const *const RSTR u_streams,
        s_t *const RSTR values
) {
  static constexpr int NC = 3;
  static constexpr int NS = 8;
  static constexpr int NDOFS = NC * NS;
  static constexpr int SHAPE_ORDER[NS] = {0, 1, 3, 2, 4, 5, 7, 6};
  const s_t *ordered_u_streams[NDOFS];
  for (int shape = 0; shape < NS; ++shape) {
    const int source_shape = SHAPE_ORDER[shape];
    for (int component = 0; component < NC; ++component) {
      ordered_u_streams[shape * NC + component] = u_streams[source_shape * NC + component];
    }
  }
  return laplace_proteus_hex8_energy_egeometry_soa<s_t, VS>(nelements, adj, det, kappa, ordered_u_streams, values);
}

template <typename s_t, int VS>
static SFEM_INLINE int laplace_hex8_energy_ecoords_soa(
        const ptrdiff_t nelements,
        const s_t *const *const RSTR coords,
        const s_t kappa,
        const s_t *const *const RSTR u_streams,
        s_t *const RSTR values
) {
  static constexpr int NC = 3;
  static constexpr int NS = 8;
  static constexpr int NDOFS = NC * NS;
  static constexpr int SHAPE_ORDER[NS] = {0, 1, 3, 2, 4, 5, 7, 6};
  const s_t *ordered_coords[NDOFS];
  for (int shape = 0; shape < NS; ++shape) {
    const int source_shape = SHAPE_ORDER[shape];
    for (int component = 0; component < NC; ++component) {
      ordered_coords[shape * NC + component] = coords[source_shape * NC + component];
    }
  }
  const s_t *ordered_u_streams[NDOFS];
  for (int shape = 0; shape < NS; ++shape) {
    const int source_shape = SHAPE_ORDER[shape];
    for (int component = 0; component < NC; ++component) {
      ordered_u_streams[shape * NC + component] = u_streams[source_shape * NC + component];
    }
  }
  return laplace_proteus_hex8_energy_ecoords_soa<s_t, VS>(nelements, ordered_coords, kappa, ordered_u_streams, values);
}

template <typename s_t, int VS>
static SFEM_INLINE int laplace_hex8_energy_esoa(
        const ptrdiff_t nelements,
        const s_t *const *const RSTR coords,
        const s_t kappa,
        const s_t *const *const RSTR u_streams,
        s_t *const RSTR values
) {
  static constexpr int NC = 3;
  static constexpr int NS = 8;
  static constexpr int NDOFS = NC * NS;
  static constexpr int SHAPE_ORDER[NS] = {0, 1, 3, 2, 4, 5, 7, 6};
  const s_t *ordered_coords[NDOFS];
  for (int shape = 0; shape < NS; ++shape) {
    const int source_shape = SHAPE_ORDER[shape];
    for (int component = 0; component < NC; ++component) {
      ordered_coords[shape * NC + component] = coords[source_shape * NC + component];
    }
  }
  const s_t *ordered_u_streams[NDOFS];
  for (int shape = 0; shape < NS; ++shape) {
    const int source_shape = SHAPE_ORDER[shape];
    for (int component = 0; component < NC; ++component) {
      ordered_u_streams[shape * NC + component] = u_streams[source_shape * NC + component];
    }
  }
  return laplace_proteus_hex8_energy_esoa<s_t, VS>(nelements, ordered_coords, kappa, ordered_u_streams, values);
}

template <typename s_t, int VS>
static SFEM_INLINE int laplace_hex8_gradient_egeometry_soa(
        const ptrdiff_t nelements,
        const s_t *const *const RSTR adj,
        const s_t *const RSTR det,
        const s_t kappa,
        const s_t *const *const RSTR u_streams,
        s_t *const *const RSTR out_streams
) {
  static constexpr int NC = 3;
  static constexpr int NS = 8;
  static constexpr int NDOFS = NC * NS;
  static constexpr int SHAPE_ORDER[NS] = {0, 1, 3, 2, 4, 5, 7, 6};
  const s_t *ordered_u_streams[NDOFS];
  for (int shape = 0; shape < NS; ++shape) {
    const int source_shape = SHAPE_ORDER[shape];
    for (int component = 0; component < NC; ++component) {
      ordered_u_streams[shape * NC + component] = u_streams[source_shape * NC + component];
    }
  }
  s_t *ordered_out_streams[NDOFS];
  for (int shape = 0; shape < NS; ++shape) {
    const int source_shape = SHAPE_ORDER[shape];
    for (int component = 0; component < NC; ++component) {
      ordered_out_streams[shape * NC + component] = out_streams[source_shape * NC + component];
    }
  }
  return laplace_proteus_hex8_gradient_egeometry_soa<s_t, VS>(nelements, adj, det, kappa, ordered_u_streams, ordered_out_streams);
}

template <typename s_t, int VS>
static SFEM_INLINE int laplace_hex8_gradient_ecoords_soa(
        const ptrdiff_t nelements,
        const s_t *const *const RSTR coords,
        const s_t kappa,
        const s_t *const *const RSTR u_streams,
        s_t *const *const RSTR out_streams
) {
  static constexpr int NC = 3;
  static constexpr int NS = 8;
  static constexpr int NDOFS = NC * NS;
  static constexpr int SHAPE_ORDER[NS] = {0, 1, 3, 2, 4, 5, 7, 6};
  const s_t *ordered_coords[NDOFS];
  for (int shape = 0; shape < NS; ++shape) {
    const int source_shape = SHAPE_ORDER[shape];
    for (int component = 0; component < NC; ++component) {
      ordered_coords[shape * NC + component] = coords[source_shape * NC + component];
    }
  }
  const s_t *ordered_u_streams[NDOFS];
  for (int shape = 0; shape < NS; ++shape) {
    const int source_shape = SHAPE_ORDER[shape];
    for (int component = 0; component < NC; ++component) {
      ordered_u_streams[shape * NC + component] = u_streams[source_shape * NC + component];
    }
  }
  s_t *ordered_out_streams[NDOFS];
  for (int shape = 0; shape < NS; ++shape) {
    const int source_shape = SHAPE_ORDER[shape];
    for (int component = 0; component < NC; ++component) {
      ordered_out_streams[shape * NC + component] = out_streams[source_shape * NC + component];
    }
  }
  return laplace_proteus_hex8_gradient_ecoords_soa<s_t, VS>(nelements, ordered_coords, kappa, ordered_u_streams, ordered_out_streams);
}

template <typename s_t, int VS>
static SFEM_INLINE int laplace_hex8_gradient_esoa(
        const ptrdiff_t nelements,
        const s_t *const *const RSTR coords,
        const s_t kappa,
        const s_t *const *const RSTR u_streams,
        s_t *const *const RSTR out_streams
) {
  static constexpr int NC = 3;
  static constexpr int NS = 8;
  static constexpr int NDOFS = NC * NS;
  static constexpr int SHAPE_ORDER[NS] = {0, 1, 3, 2, 4, 5, 7, 6};
  const s_t *ordered_coords[NDOFS];
  for (int shape = 0; shape < NS; ++shape) {
    const int source_shape = SHAPE_ORDER[shape];
    for (int component = 0; component < NC; ++component) {
      ordered_coords[shape * NC + component] = coords[source_shape * NC + component];
    }
  }
  const s_t *ordered_u_streams[NDOFS];
  for (int shape = 0; shape < NS; ++shape) {
    const int source_shape = SHAPE_ORDER[shape];
    for (int component = 0; component < NC; ++component) {
      ordered_u_streams[shape * NC + component] = u_streams[source_shape * NC + component];
    }
  }
  s_t *ordered_out_streams[NDOFS];
  for (int shape = 0; shape < NS; ++shape) {
    const int source_shape = SHAPE_ORDER[shape];
    for (int component = 0; component < NC; ++component) {
      ordered_out_streams[shape * NC + component] = out_streams[source_shape * NC + component];
    }
  }
  return laplace_proteus_hex8_gradient_esoa<s_t, VS>(nelements, ordered_coords, kappa, ordered_u_streams, ordered_out_streams);
}

template <typename s_t, int VS>
static SFEM_INLINE int laplace_hex8_hessian_egeometry_soa(
        const ptrdiff_t nelements,
        const s_t *const *const RSTR adj,
        const s_t *const RSTR det,
        const s_t kappa,
        s_t *const *const RSTR matrix_streams
) {
  static constexpr int NC = 3;
  static constexpr int NS = 8;
  static constexpr int NDOFS = NC * NS;
  static constexpr int SHAPE_ORDER[NS] = {0, 1, 3, 2, 4, 5, 7, 6};
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
  return laplace_proteus_hex8_hessian_egeometry_soa<s_t, VS>(nelements, adj, det, kappa, ordered_matrix_streams);
}

template <typename s_t, int VS>
static SFEM_INLINE int laplace_hex8_hessian_ecoords_soa(
        const ptrdiff_t nelements,
        const s_t *const *const RSTR coords,
        const s_t kappa,
        s_t *const *const RSTR matrix_streams
) {
  static constexpr int NC = 3;
  static constexpr int NS = 8;
  static constexpr int NDOFS = NC * NS;
  static constexpr int SHAPE_ORDER[NS] = {0, 1, 3, 2, 4, 5, 7, 6};
  const s_t *ordered_coords[NDOFS];
  for (int shape = 0; shape < NS; ++shape) {
    const int source_shape = SHAPE_ORDER[shape];
    for (int component = 0; component < NC; ++component) {
      ordered_coords[shape * NC + component] = coords[source_shape * NC + component];
    }
  }
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
  return laplace_proteus_hex8_hessian_ecoords_soa<s_t, VS>(nelements, ordered_coords, kappa, ordered_matrix_streams);
}

template <typename s_t, int VS>
static SFEM_INLINE int laplace_hex8_hessian_esoa(
        const ptrdiff_t nelements,
        const s_t *const *const RSTR coords,
        const s_t kappa,
        s_t *const *const RSTR matrix_streams
) {
  static constexpr int NC = 3;
  static constexpr int NS = 8;
  static constexpr int NDOFS = NC * NS;
  static constexpr int SHAPE_ORDER[NS] = {0, 1, 3, 2, 4, 5, 7, 6};
  const s_t *ordered_coords[NDOFS];
  for (int shape = 0; shape < NS; ++shape) {
    const int source_shape = SHAPE_ORDER[shape];
    for (int component = 0; component < NC; ++component) {
      ordered_coords[shape * NC + component] = coords[source_shape * NC + component];
    }
  }
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
  return laplace_proteus_hex8_hessian_esoa<s_t, VS>(nelements, ordered_coords, kappa, ordered_matrix_streams);
}

} // namespace codegen
} // namespace sfem

#endif
