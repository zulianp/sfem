#ifndef MODIFIED_MOONEY_RIVLIN_QUAD4_ELEMENT_API_HPP
#define MODIFIED_MOONEY_RIVLIN_QUAD4_ELEMENT_API_HPP

#include "../proteus_quad4/modified_mooney_rivlin_proteus_quad4_element.hpp"

namespace sfem {
namespace codegen {

template <typename scalar_t, int VECTOR_SIZE = 16>
static SFEM_INLINE int modified_mooney_rivlin_quad4_energy_element_geometry_soa(
        const ptrdiff_t nelements,
        const scalar_t *const *const SFEM_RESTRICT jacobian_adjugate,
        const scalar_t *const SFEM_RESTRICT jacobian_determinant,
        const scalar_t c1,
        const scalar_t c2,
        const scalar_t kappa,
        const scalar_t *const *const SFEM_RESTRICT u_streams,
        scalar_t *const SFEM_RESTRICT values
) {
    static constexpr int DIM = 2;
    static constexpr int N_SHAPE = 4;
    static constexpr int N_QP = 4;
    static constexpr int NDOFS = DIM * N_SHAPE;
    static constexpr int SHAPE_ORDER[N_SHAPE] = {0, 1, 3, 2};
    const scalar_t *ordered_u_streams[NDOFS];
    for (int shape = 0; shape < N_SHAPE; ++shape) {
        const int source_shape = SHAPE_ORDER[shape];
        for (int component = 0; component < DIM; ++component) {
            ordered_u_streams[shape * DIM + component] = u_streams[source_shape * DIM + component];
        }
    }
    return modified_mooney_rivlin_proteus_quad4_energy_element_geometry_soa<scalar_t, VECTOR_SIZE>(nelements, jacobian_adjugate, jacobian_determinant, c1, c2, kappa, ordered_u_streams, values);
}

template <typename scalar_t, int VECTOR_SIZE = 16>
static SFEM_INLINE int modified_mooney_rivlin_quad4_energy_element_coords_soa(
        const ptrdiff_t nelements,
        const scalar_t *const *const SFEM_RESTRICT coords,
        const scalar_t c1,
        const scalar_t c2,
        const scalar_t kappa,
        const scalar_t *const *const SFEM_RESTRICT u_streams,
        scalar_t *const SFEM_RESTRICT values
) {
    static constexpr int DIM = 2;
    static constexpr int N_SHAPE = 4;
    static constexpr int N_QP = 4;
    static constexpr int NDOFS = DIM * N_SHAPE;
    static constexpr int SHAPE_ORDER[N_SHAPE] = {0, 1, 3, 2};
    const scalar_t *ordered_coords[NDOFS];
    for (int shape = 0; shape < N_SHAPE; ++shape) {
        const int source_shape = SHAPE_ORDER[shape];
        for (int component = 0; component < DIM; ++component) {
            ordered_coords[shape * DIM + component] = coords[source_shape * DIM + component];
        }
    }
    const scalar_t *ordered_u_streams[NDOFS];
    for (int shape = 0; shape < N_SHAPE; ++shape) {
        const int source_shape = SHAPE_ORDER[shape];
        for (int component = 0; component < DIM; ++component) {
            ordered_u_streams[shape * DIM + component] = u_streams[source_shape * DIM + component];
        }
    }
    return modified_mooney_rivlin_proteus_quad4_energy_element_coords_soa<scalar_t, VECTOR_SIZE>(nelements, ordered_coords, c1, c2, kappa, ordered_u_streams, values);
}

template <typename scalar_t, int VECTOR_SIZE = 16>
static SFEM_INLINE int modified_mooney_rivlin_quad4_energy_element_soa(
        const ptrdiff_t nelements,
        const scalar_t *const *const SFEM_RESTRICT coords,
        const scalar_t c1,
        const scalar_t c2,
        const scalar_t kappa,
        const scalar_t *const *const SFEM_RESTRICT u_streams,
        scalar_t *const SFEM_RESTRICT values
) {
    static constexpr int DIM = 2;
    static constexpr int N_SHAPE = 4;
    static constexpr int N_QP = 4;
    static constexpr int NDOFS = DIM * N_SHAPE;
    static constexpr int SHAPE_ORDER[N_SHAPE] = {0, 1, 3, 2};
    const scalar_t *ordered_coords[NDOFS];
    for (int shape = 0; shape < N_SHAPE; ++shape) {
        const int source_shape = SHAPE_ORDER[shape];
        for (int component = 0; component < DIM; ++component) {
            ordered_coords[shape * DIM + component] = coords[source_shape * DIM + component];
        }
    }
    const scalar_t *ordered_u_streams[NDOFS];
    for (int shape = 0; shape < N_SHAPE; ++shape) {
        const int source_shape = SHAPE_ORDER[shape];
        for (int component = 0; component < DIM; ++component) {
            ordered_u_streams[shape * DIM + component] = u_streams[source_shape * DIM + component];
        }
    }
    return modified_mooney_rivlin_proteus_quad4_energy_element_soa<scalar_t, VECTOR_SIZE>(nelements, ordered_coords, c1, c2, kappa, ordered_u_streams, values);
}

template <typename scalar_t, int VECTOR_SIZE = 16>
static SFEM_INLINE int modified_mooney_rivlin_quad4_gradient_element_geometry_soa(
        const ptrdiff_t nelements,
        const scalar_t *const *const SFEM_RESTRICT jacobian_adjugate,
        const scalar_t *const SFEM_RESTRICT jacobian_determinant,
        const scalar_t c1,
        const scalar_t c2,
        const scalar_t kappa,
        const scalar_t *const *const SFEM_RESTRICT u_streams,
        scalar_t *const *const SFEM_RESTRICT out_streams
) {
    static constexpr int DIM = 2;
    static constexpr int N_SHAPE = 4;
    static constexpr int N_QP = 4;
    static constexpr int NDOFS = DIM * N_SHAPE;
    static constexpr int SHAPE_ORDER[N_SHAPE] = {0, 1, 3, 2};
    const scalar_t *ordered_u_streams[NDOFS];
    for (int shape = 0; shape < N_SHAPE; ++shape) {
        const int source_shape = SHAPE_ORDER[shape];
        for (int component = 0; component < DIM; ++component) {
            ordered_u_streams[shape * DIM + component] = u_streams[source_shape * DIM + component];
        }
    }
    scalar_t *ordered_out_streams[NDOFS];
    for (int shape = 0; shape < N_SHAPE; ++shape) {
        const int source_shape = SHAPE_ORDER[shape];
        for (int component = 0; component < DIM; ++component) {
            ordered_out_streams[shape * DIM + component] = out_streams[source_shape * DIM + component];
        }
    }
    return modified_mooney_rivlin_proteus_quad4_gradient_element_geometry_soa<scalar_t, VECTOR_SIZE>(nelements, jacobian_adjugate, jacobian_determinant, c1, c2, kappa, ordered_u_streams, ordered_out_streams);
}

template <typename scalar_t, int VECTOR_SIZE = 16>
static SFEM_INLINE int modified_mooney_rivlin_quad4_gradient_element_coords_soa(
        const ptrdiff_t nelements,
        const scalar_t *const *const SFEM_RESTRICT coords,
        const scalar_t c1,
        const scalar_t c2,
        const scalar_t kappa,
        const scalar_t *const *const SFEM_RESTRICT u_streams,
        scalar_t *const *const SFEM_RESTRICT out_streams
) {
    static constexpr int DIM = 2;
    static constexpr int N_SHAPE = 4;
    static constexpr int N_QP = 4;
    static constexpr int NDOFS = DIM * N_SHAPE;
    static constexpr int SHAPE_ORDER[N_SHAPE] = {0, 1, 3, 2};
    const scalar_t *ordered_coords[NDOFS];
    for (int shape = 0; shape < N_SHAPE; ++shape) {
        const int source_shape = SHAPE_ORDER[shape];
        for (int component = 0; component < DIM; ++component) {
            ordered_coords[shape * DIM + component] = coords[source_shape * DIM + component];
        }
    }
    const scalar_t *ordered_u_streams[NDOFS];
    for (int shape = 0; shape < N_SHAPE; ++shape) {
        const int source_shape = SHAPE_ORDER[shape];
        for (int component = 0; component < DIM; ++component) {
            ordered_u_streams[shape * DIM + component] = u_streams[source_shape * DIM + component];
        }
    }
    scalar_t *ordered_out_streams[NDOFS];
    for (int shape = 0; shape < N_SHAPE; ++shape) {
        const int source_shape = SHAPE_ORDER[shape];
        for (int component = 0; component < DIM; ++component) {
            ordered_out_streams[shape * DIM + component] = out_streams[source_shape * DIM + component];
        }
    }
    return modified_mooney_rivlin_proteus_quad4_gradient_element_coords_soa<scalar_t, VECTOR_SIZE>(nelements, ordered_coords, c1, c2, kappa, ordered_u_streams, ordered_out_streams);
}

template <typename scalar_t, int VECTOR_SIZE = 16>
static SFEM_INLINE int modified_mooney_rivlin_quad4_gradient_element_soa(
        const ptrdiff_t nelements,
        const scalar_t *const *const SFEM_RESTRICT coords,
        const scalar_t c1,
        const scalar_t c2,
        const scalar_t kappa,
        const scalar_t *const *const SFEM_RESTRICT u_streams,
        scalar_t *const *const SFEM_RESTRICT out_streams
) {
    static constexpr int DIM = 2;
    static constexpr int N_SHAPE = 4;
    static constexpr int N_QP = 4;
    static constexpr int NDOFS = DIM * N_SHAPE;
    static constexpr int SHAPE_ORDER[N_SHAPE] = {0, 1, 3, 2};
    const scalar_t *ordered_coords[NDOFS];
    for (int shape = 0; shape < N_SHAPE; ++shape) {
        const int source_shape = SHAPE_ORDER[shape];
        for (int component = 0; component < DIM; ++component) {
            ordered_coords[shape * DIM + component] = coords[source_shape * DIM + component];
        }
    }
    const scalar_t *ordered_u_streams[NDOFS];
    for (int shape = 0; shape < N_SHAPE; ++shape) {
        const int source_shape = SHAPE_ORDER[shape];
        for (int component = 0; component < DIM; ++component) {
            ordered_u_streams[shape * DIM + component] = u_streams[source_shape * DIM + component];
        }
    }
    scalar_t *ordered_out_streams[NDOFS];
    for (int shape = 0; shape < N_SHAPE; ++shape) {
        const int source_shape = SHAPE_ORDER[shape];
        for (int component = 0; component < DIM; ++component) {
            ordered_out_streams[shape * DIM + component] = out_streams[source_shape * DIM + component];
        }
    }
    return modified_mooney_rivlin_proteus_quad4_gradient_element_soa<scalar_t, VECTOR_SIZE>(nelements, ordered_coords, c1, c2, kappa, ordered_u_streams, ordered_out_streams);
}

template <typename scalar_t, int VECTOR_SIZE = 16>
static SFEM_INLINE int modified_mooney_rivlin_quad4_hessian_element_geometry_soa(
        const ptrdiff_t nelements,
        const scalar_t *const *const SFEM_RESTRICT jacobian_adjugate,
        const scalar_t *const SFEM_RESTRICT jacobian_determinant,
        const scalar_t c1,
        const scalar_t c2,
        const scalar_t kappa,
        const scalar_t *const *const SFEM_RESTRICT u_streams,
        scalar_t *const *const SFEM_RESTRICT matrix_streams
) {
    static constexpr int DIM = 2;
    static constexpr int N_SHAPE = 4;
    static constexpr int N_QP = 4;
    static constexpr int NDOFS = DIM * N_SHAPE;
    static constexpr int SHAPE_ORDER[N_SHAPE] = {0, 1, 3, 2};
    const scalar_t *ordered_u_streams[NDOFS];
    for (int shape = 0; shape < N_SHAPE; ++shape) {
        const int source_shape = SHAPE_ORDER[shape];
        for (int component = 0; component < DIM; ++component) {
            ordered_u_streams[shape * DIM + component] = u_streams[source_shape * DIM + component];
        }
    }
    scalar_t *ordered_matrix_streams[NDOFS * NDOFS];
    for (int row_shape = 0; row_shape < N_SHAPE; ++row_shape) {
        const int source_row_shape = SHAPE_ORDER[row_shape];
        for (int row_component = 0; row_component < DIM; ++row_component) {
            const int row = row_shape * DIM + row_component;
            const int source_row = source_row_shape * DIM + row_component;
            for (int col_shape = 0; col_shape < N_SHAPE; ++col_shape) {
                const int source_col_shape = SHAPE_ORDER[col_shape];
                for (int col_component = 0; col_component < DIM; ++col_component) {
                    const int col = col_shape * DIM + col_component;
                    const int source_col = source_col_shape * DIM + col_component;
                    ordered_matrix_streams[row * NDOFS + col] = matrix_streams[source_row * NDOFS + source_col];
                }
            }
        }
    }
    return modified_mooney_rivlin_proteus_quad4_hessian_element_geometry_soa<scalar_t, VECTOR_SIZE>(nelements, jacobian_adjugate, jacobian_determinant, c1, c2, kappa, ordered_u_streams, ordered_matrix_streams);
}

template <typename scalar_t, int VECTOR_SIZE = 16>
static SFEM_INLINE int modified_mooney_rivlin_quad4_hessian_element_coords_soa(
        const ptrdiff_t nelements,
        const scalar_t *const *const SFEM_RESTRICT coords,
        const scalar_t c1,
        const scalar_t c2,
        const scalar_t kappa,
        const scalar_t *const *const SFEM_RESTRICT u_streams,
        scalar_t *const *const SFEM_RESTRICT matrix_streams
) {
    static constexpr int DIM = 2;
    static constexpr int N_SHAPE = 4;
    static constexpr int N_QP = 4;
    static constexpr int NDOFS = DIM * N_SHAPE;
    static constexpr int SHAPE_ORDER[N_SHAPE] = {0, 1, 3, 2};
    const scalar_t *ordered_coords[NDOFS];
    for (int shape = 0; shape < N_SHAPE; ++shape) {
        const int source_shape = SHAPE_ORDER[shape];
        for (int component = 0; component < DIM; ++component) {
            ordered_coords[shape * DIM + component] = coords[source_shape * DIM + component];
        }
    }
    const scalar_t *ordered_u_streams[NDOFS];
    for (int shape = 0; shape < N_SHAPE; ++shape) {
        const int source_shape = SHAPE_ORDER[shape];
        for (int component = 0; component < DIM; ++component) {
            ordered_u_streams[shape * DIM + component] = u_streams[source_shape * DIM + component];
        }
    }
    scalar_t *ordered_matrix_streams[NDOFS * NDOFS];
    for (int row_shape = 0; row_shape < N_SHAPE; ++row_shape) {
        const int source_row_shape = SHAPE_ORDER[row_shape];
        for (int row_component = 0; row_component < DIM; ++row_component) {
            const int row = row_shape * DIM + row_component;
            const int source_row = source_row_shape * DIM + row_component;
            for (int col_shape = 0; col_shape < N_SHAPE; ++col_shape) {
                const int source_col_shape = SHAPE_ORDER[col_shape];
                for (int col_component = 0; col_component < DIM; ++col_component) {
                    const int col = col_shape * DIM + col_component;
                    const int source_col = source_col_shape * DIM + col_component;
                    ordered_matrix_streams[row * NDOFS + col] = matrix_streams[source_row * NDOFS + source_col];
                }
            }
        }
    }
    return modified_mooney_rivlin_proteus_quad4_hessian_element_coords_soa<scalar_t, VECTOR_SIZE>(nelements, ordered_coords, c1, c2, kappa, ordered_u_streams, ordered_matrix_streams);
}

template <typename scalar_t, int VECTOR_SIZE = 16>
static SFEM_INLINE int modified_mooney_rivlin_quad4_hessian_element_soa(
        const ptrdiff_t nelements,
        const scalar_t *const *const SFEM_RESTRICT coords,
        const scalar_t c1,
        const scalar_t c2,
        const scalar_t kappa,
        const scalar_t *const *const SFEM_RESTRICT u_streams,
        scalar_t *const *const SFEM_RESTRICT matrix_streams
) {
    static constexpr int DIM = 2;
    static constexpr int N_SHAPE = 4;
    static constexpr int N_QP = 4;
    static constexpr int NDOFS = DIM * N_SHAPE;
    static constexpr int SHAPE_ORDER[N_SHAPE] = {0, 1, 3, 2};
    const scalar_t *ordered_coords[NDOFS];
    for (int shape = 0; shape < N_SHAPE; ++shape) {
        const int source_shape = SHAPE_ORDER[shape];
        for (int component = 0; component < DIM; ++component) {
            ordered_coords[shape * DIM + component] = coords[source_shape * DIM + component];
        }
    }
    const scalar_t *ordered_u_streams[NDOFS];
    for (int shape = 0; shape < N_SHAPE; ++shape) {
        const int source_shape = SHAPE_ORDER[shape];
        for (int component = 0; component < DIM; ++component) {
            ordered_u_streams[shape * DIM + component] = u_streams[source_shape * DIM + component];
        }
    }
    scalar_t *ordered_matrix_streams[NDOFS * NDOFS];
    for (int row_shape = 0; row_shape < N_SHAPE; ++row_shape) {
        const int source_row_shape = SHAPE_ORDER[row_shape];
        for (int row_component = 0; row_component < DIM; ++row_component) {
            const int row = row_shape * DIM + row_component;
            const int source_row = source_row_shape * DIM + row_component;
            for (int col_shape = 0; col_shape < N_SHAPE; ++col_shape) {
                const int source_col_shape = SHAPE_ORDER[col_shape];
                for (int col_component = 0; col_component < DIM; ++col_component) {
                    const int col = col_shape * DIM + col_component;
                    const int source_col = source_col_shape * DIM + col_component;
                    ordered_matrix_streams[row * NDOFS + col] = matrix_streams[source_row * NDOFS + source_col];
                }
            }
        }
    }
    return modified_mooney_rivlin_proteus_quad4_hessian_element_soa<scalar_t, VECTOR_SIZE>(nelements, ordered_coords, c1, c2, kappa, ordered_u_streams, ordered_matrix_streams);
}

} // namespace codegen
} // namespace sfem

#endif
