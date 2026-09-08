#ifndef NEOHOOKEAN_OGDEN_QUAD4_ELEMENT_API_HPP
#define NEOHOOKEAN_OGDEN_QUAD4_ELEMENT_API_HPP

#include "../proteus_quad4/neohookean_ogden_proteus_quad4_element.hpp"

namespace sfem {
namespace codegen {

template <typename s_t, int VS = 16>
static SFEM_INLINE int neohookean_ogden_quad4_energy_element_geometry_soa(
        const ptrdiff_t nelements,
        const s_t *const *const SFEM_RESTRICT jacobian_adjugate,
        const s_t *const SFEM_RESTRICT jacobian_determinant,
        const s_t lmbda,
        const s_t mu,
        const s_t *const *const SFEM_RESTRICT u_streams,
        s_t *const SFEM_RESTRICT values
) {
    static constexpr int NC = 2;
    static constexpr int ND = 2;
    static constexpr int NS = 4;
    static constexpr int NQ = 4;
    static constexpr int NDOFS = NC * NS;
    static constexpr int SHAPE_ORDER[NS] = {0, 1, 3, 2};
    const s_t *ordered_u_streams[NDOFS];
    for (int shape = 0; shape < NS; ++shape) {
        const int source_shape = SHAPE_ORDER[shape];
        for (int component = 0; component < NC; ++component) {
            ordered_u_streams[shape * NC + component] = u_streams[source_shape * NC + component];
        }
    }
    return neohookean_ogden_proteus_quad4_energy_element_geometry_soa<s_t, VS>(nelements, jacobian_adjugate, jacobian_determinant, lmbda, mu, ordered_u_streams, values);
}

template <typename s_t, int VS = 16>
static SFEM_INLINE int neohookean_ogden_quad4_energy_element_coords_soa(
        const ptrdiff_t nelements,
        const s_t *const *const SFEM_RESTRICT coords,
        const s_t lmbda,
        const s_t mu,
        const s_t *const *const SFEM_RESTRICT u_streams,
        s_t *const SFEM_RESTRICT values
) {
    static constexpr int NC = 2;
    static constexpr int ND = 2;
    static constexpr int NS = 4;
    static constexpr int NQ = 4;
    static constexpr int NDOFS = NC * NS;
    static constexpr int SHAPE_ORDER[NS] = {0, 1, 3, 2};
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
    return neohookean_ogden_proteus_quad4_energy_element_coords_soa<s_t, VS>(nelements, ordered_coords, lmbda, mu, ordered_u_streams, values);
}

template <typename s_t, int VS = 16>
static SFEM_INLINE int neohookean_ogden_quad4_energy_element_soa(
        const ptrdiff_t nelements,
        const s_t *const *const SFEM_RESTRICT coords,
        const s_t lmbda,
        const s_t mu,
        const s_t *const *const SFEM_RESTRICT u_streams,
        s_t *const SFEM_RESTRICT values
) {
    static constexpr int NC = 2;
    static constexpr int ND = 2;
    static constexpr int NS = 4;
    static constexpr int NQ = 4;
    static constexpr int NDOFS = NC * NS;
    static constexpr int SHAPE_ORDER[NS] = {0, 1, 3, 2};
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
    return neohookean_ogden_proteus_quad4_energy_element_soa<s_t, VS>(nelements, ordered_coords, lmbda, mu, ordered_u_streams, values);
}

template <typename s_t, int VS = 16>
static SFEM_INLINE int neohookean_ogden_quad4_gradient_element_geometry_soa(
        const ptrdiff_t nelements,
        const s_t *const *const SFEM_RESTRICT jacobian_adjugate,
        const s_t *const SFEM_RESTRICT jacobian_determinant,
        const s_t lmbda,
        const s_t mu,
        const s_t *const *const SFEM_RESTRICT u_streams,
        s_t *const *const SFEM_RESTRICT out_streams
) {
    static constexpr int NC = 2;
    static constexpr int ND = 2;
    static constexpr int NS = 4;
    static constexpr int NQ = 4;
    static constexpr int NDOFS = NC * NS;
    static constexpr int SHAPE_ORDER[NS] = {0, 1, 3, 2};
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
    return neohookean_ogden_proteus_quad4_gradient_element_geometry_soa<s_t, VS>(nelements, jacobian_adjugate, jacobian_determinant, lmbda, mu, ordered_u_streams, ordered_out_streams);
}

template <typename s_t, int VS = 16>
static SFEM_INLINE int neohookean_ogden_quad4_gradient_element_coords_soa(
        const ptrdiff_t nelements,
        const s_t *const *const SFEM_RESTRICT coords,
        const s_t lmbda,
        const s_t mu,
        const s_t *const *const SFEM_RESTRICT u_streams,
        s_t *const *const SFEM_RESTRICT out_streams
) {
    static constexpr int NC = 2;
    static constexpr int ND = 2;
    static constexpr int NS = 4;
    static constexpr int NQ = 4;
    static constexpr int NDOFS = NC * NS;
    static constexpr int SHAPE_ORDER[NS] = {0, 1, 3, 2};
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
    return neohookean_ogden_proteus_quad4_gradient_element_coords_soa<s_t, VS>(nelements, ordered_coords, lmbda, mu, ordered_u_streams, ordered_out_streams);
}

template <typename s_t, int VS = 16>
static SFEM_INLINE int neohookean_ogden_quad4_gradient_element_soa(
        const ptrdiff_t nelements,
        const s_t *const *const SFEM_RESTRICT coords,
        const s_t lmbda,
        const s_t mu,
        const s_t *const *const SFEM_RESTRICT u_streams,
        s_t *const *const SFEM_RESTRICT out_streams
) {
    static constexpr int NC = 2;
    static constexpr int ND = 2;
    static constexpr int NS = 4;
    static constexpr int NQ = 4;
    static constexpr int NDOFS = NC * NS;
    static constexpr int SHAPE_ORDER[NS] = {0, 1, 3, 2};
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
    return neohookean_ogden_proteus_quad4_gradient_element_soa<s_t, VS>(nelements, ordered_coords, lmbda, mu, ordered_u_streams, ordered_out_streams);
}

template <typename s_t, int VS = 16>
static SFEM_INLINE int neohookean_ogden_quad4_hessian_element_geometry_soa(
        const ptrdiff_t nelements,
        const s_t *const *const SFEM_RESTRICT jacobian_adjugate,
        const s_t *const SFEM_RESTRICT jacobian_determinant,
        const s_t lmbda,
        const s_t mu,
        const s_t *const *const SFEM_RESTRICT u_streams,
        s_t *const *const SFEM_RESTRICT matrix_streams
) {
    static constexpr int NC = 2;
    static constexpr int ND = 2;
    static constexpr int NS = 4;
    static constexpr int NQ = 4;
    static constexpr int NDOFS = NC * NS;
    static constexpr int SHAPE_ORDER[NS] = {0, 1, 3, 2};
    const s_t *ordered_u_streams[NDOFS];
    for (int shape = 0; shape < NS; ++shape) {
        const int source_shape = SHAPE_ORDER[shape];
        for (int component = 0; component < NC; ++component) {
            ordered_u_streams[shape * NC + component] = u_streams[source_shape * NC + component];
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
    return neohookean_ogden_proteus_quad4_hessian_element_geometry_soa<s_t, VS>(nelements, jacobian_adjugate, jacobian_determinant, lmbda, mu, ordered_u_streams, ordered_matrix_streams);
}

template <typename s_t, int VS = 16>
static SFEM_INLINE int neohookean_ogden_quad4_hessian_element_coords_soa(
        const ptrdiff_t nelements,
        const s_t *const *const SFEM_RESTRICT coords,
        const s_t lmbda,
        const s_t mu,
        const s_t *const *const SFEM_RESTRICT u_streams,
        s_t *const *const SFEM_RESTRICT matrix_streams
) {
    static constexpr int NC = 2;
    static constexpr int ND = 2;
    static constexpr int NS = 4;
    static constexpr int NQ = 4;
    static constexpr int NDOFS = NC * NS;
    static constexpr int SHAPE_ORDER[NS] = {0, 1, 3, 2};
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
    return neohookean_ogden_proteus_quad4_hessian_element_coords_soa<s_t, VS>(nelements, ordered_coords, lmbda, mu, ordered_u_streams, ordered_matrix_streams);
}

template <typename s_t, int VS = 16>
static SFEM_INLINE int neohookean_ogden_quad4_hessian_element_soa(
        const ptrdiff_t nelements,
        const s_t *const *const SFEM_RESTRICT coords,
        const s_t lmbda,
        const s_t mu,
        const s_t *const *const SFEM_RESTRICT u_streams,
        s_t *const *const SFEM_RESTRICT matrix_streams
) {
    static constexpr int NC = 2;
    static constexpr int ND = 2;
    static constexpr int NS = 4;
    static constexpr int NQ = 4;
    static constexpr int NDOFS = NC * NS;
    static constexpr int SHAPE_ORDER[NS] = {0, 1, 3, 2};
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
    return neohookean_ogden_proteus_quad4_hessian_element_soa<s_t, VS>(nelements, ordered_coords, lmbda, mu, ordered_u_streams, ordered_matrix_streams);
}

} // namespace codegen
} // namespace sfem

#endif
