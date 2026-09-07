#pragma once

#include <cstddef>

#ifndef SFEM_SUCCESS
#define SFEM_SUCCESS 0
#endif
#ifndef SFEM_FAILURE
#define SFEM_FAILURE 1
#endif

#include "../d2/proteus_quad4/neohookean_ogden_proteus_quad4_element.hpp"
#include "../d2/quad4/neohookean_ogden_quad4_element.hpp"
#include "../d2/tri3/neohookean_ogden_tri3_element.hpp"
#include "../d3/hex8/neohookean_ogden_hex8_element.hpp"
#include "../d3/proteus_hex8/neohookean_ogden_proteus_hex8_element.hpp"
#include "../d3/tet10/neohookean_ogden_tet10_element.hpp"
#include "../d3/tet4/neohookean_ogden_tet4_element.hpp"

namespace sfem {
namespace codegen {

template <typename scalar_t, int VECTOR_SIZE = 16, typename elem_type_t>
static SFEM_INLINE int neohookean_ogden_energy_2d_element_soa(
        const elem_type_t element_type,
        const ptrdiff_t nelements,
        const scalar_t *const *const SFEM_RESTRICT coords,
        const scalar_t lmbda,
        const scalar_t mu,
        const scalar_t *const *const SFEM_RESTRICT u_streams,
        scalar_t *const SFEM_RESTRICT values
) {
    switch ((int)element_type) {
        case 3:
            return neohookean_ogden_tri3_energy_element_soa<scalar_t, VECTOR_SIZE>(nelements, coords, lmbda, mu, u_streams, values);
        case 40:
            return neohookean_ogden_quad4_energy_element_soa<scalar_t, VECTOR_SIZE>(nelements, coords, lmbda, mu, u_streams, values);
        case 400000:
            return neohookean_ogden_proteus_quad4_energy_element_soa<scalar_t, VECTOR_SIZE>(nelements, coords, lmbda, mu, u_streams, values);
        default:
            return SFEM_FAILURE;
    }
}

template <typename scalar_t, int VECTOR_SIZE = 16, typename elem_type_t>
static SFEM_INLINE int neohookean_ogden_energy_3d_element_soa(
        const elem_type_t element_type,
        const ptrdiff_t nelements,
        const scalar_t *const *const SFEM_RESTRICT coords,
        const scalar_t lmbda,
        const scalar_t mu,
        const scalar_t *const *const SFEM_RESTRICT u_streams,
        scalar_t *const SFEM_RESTRICT values
) {
    switch ((int)element_type) {
        case 4:
            return neohookean_ogden_tet4_energy_element_soa<scalar_t, VECTOR_SIZE>(nelements, coords, lmbda, mu, u_streams, values);
        case 10:
            return neohookean_ogden_tet10_energy_element_soa<scalar_t, VECTOR_SIZE>(nelements, coords, lmbda, mu, u_streams, values);
        case 8:
            return neohookean_ogden_hex8_energy_element_soa<scalar_t, VECTOR_SIZE>(nelements, coords, lmbda, mu, u_streams, values);
        case 100008:
            return neohookean_ogden_proteus_hex8_energy_element_soa<scalar_t, VECTOR_SIZE>(nelements, coords, lmbda, mu, u_streams, values);
        default:
            return SFEM_FAILURE;
    }
}

template <typename scalar_t, int VECTOR_SIZE = 16, typename elem_type_t>
static SFEM_INLINE int neohookean_ogden_energy_2d_element_coords_soa(
        const elem_type_t element_type,
        const ptrdiff_t nelements,
        const scalar_t *const *const SFEM_RESTRICT coords,
        const scalar_t lmbda,
        const scalar_t mu,
        const scalar_t *const *const SFEM_RESTRICT u_streams,
        scalar_t *const SFEM_RESTRICT values
) {
    switch ((int)element_type) {
        case 3:
            return neohookean_ogden_tri3_energy_element_coords_soa<scalar_t, VECTOR_SIZE>(nelements, coords, lmbda, mu, u_streams, values);
        case 40:
            return neohookean_ogden_quad4_energy_element_coords_soa<scalar_t, VECTOR_SIZE>(nelements, coords, lmbda, mu, u_streams, values);
        case 400000:
            return neohookean_ogden_proteus_quad4_energy_element_coords_soa<scalar_t, VECTOR_SIZE>(nelements, coords, lmbda, mu, u_streams, values);
        default:
            return SFEM_FAILURE;
    }
}

template <typename scalar_t, int VECTOR_SIZE = 16, typename elem_type_t>
static SFEM_INLINE int neohookean_ogden_energy_3d_element_coords_soa(
        const elem_type_t element_type,
        const ptrdiff_t nelements,
        const scalar_t *const *const SFEM_RESTRICT coords,
        const scalar_t lmbda,
        const scalar_t mu,
        const scalar_t *const *const SFEM_RESTRICT u_streams,
        scalar_t *const SFEM_RESTRICT values
) {
    switch ((int)element_type) {
        case 4:
            return neohookean_ogden_tet4_energy_element_coords_soa<scalar_t, VECTOR_SIZE>(nelements, coords, lmbda, mu, u_streams, values);
        case 10:
            return neohookean_ogden_tet10_energy_element_coords_soa<scalar_t, VECTOR_SIZE>(nelements, coords, lmbda, mu, u_streams, values);
        case 8:
            return neohookean_ogden_hex8_energy_element_coords_soa<scalar_t, VECTOR_SIZE>(nelements, coords, lmbda, mu, u_streams, values);
        case 100008:
            return neohookean_ogden_proteus_hex8_energy_element_coords_soa<scalar_t, VECTOR_SIZE>(nelements, coords, lmbda, mu, u_streams, values);
        default:
            return SFEM_FAILURE;
    }
}

template <typename scalar_t, int VECTOR_SIZE = 16, typename elem_type_t>
static SFEM_INLINE int neohookean_ogden_energy_2d_element_geometry_soa(
        const elem_type_t element_type,
        const ptrdiff_t nelements,
        const scalar_t *const *const SFEM_RESTRICT jacobian_adjugate,
        const scalar_t *const SFEM_RESTRICT jacobian_determinant,
        const scalar_t lmbda,
        const scalar_t mu,
        const scalar_t *const *const SFEM_RESTRICT u_streams,
        scalar_t *const SFEM_RESTRICT values
) {
    switch ((int)element_type) {
        case 3:
            return neohookean_ogden_tri3_energy_element_geometry_soa<scalar_t, VECTOR_SIZE>(nelements, jacobian_adjugate, jacobian_determinant, lmbda, mu, u_streams, values);
        case 40:
            return neohookean_ogden_quad4_energy_element_geometry_soa<scalar_t, VECTOR_SIZE>(nelements, jacobian_adjugate, jacobian_determinant, lmbda, mu, u_streams, values);
        case 400000:
            return neohookean_ogden_proteus_quad4_energy_element_geometry_soa<scalar_t, VECTOR_SIZE>(nelements, jacobian_adjugate, jacobian_determinant, lmbda, mu, u_streams, values);
        default:
            return SFEM_FAILURE;
    }
}

template <typename scalar_t, int VECTOR_SIZE = 16, typename elem_type_t>
static SFEM_INLINE int neohookean_ogden_energy_3d_element_geometry_soa(
        const elem_type_t element_type,
        const ptrdiff_t nelements,
        const scalar_t *const *const SFEM_RESTRICT jacobian_adjugate,
        const scalar_t *const SFEM_RESTRICT jacobian_determinant,
        const scalar_t lmbda,
        const scalar_t mu,
        const scalar_t *const *const SFEM_RESTRICT u_streams,
        scalar_t *const SFEM_RESTRICT values
) {
    switch ((int)element_type) {
        case 4:
            return neohookean_ogden_tet4_energy_element_geometry_soa<scalar_t, VECTOR_SIZE>(nelements, jacobian_adjugate, jacobian_determinant, lmbda, mu, u_streams, values);
        case 10:
            return neohookean_ogden_tet10_energy_element_geometry_soa<scalar_t, VECTOR_SIZE>(nelements, jacobian_adjugate, jacobian_determinant, lmbda, mu, u_streams, values);
        case 8:
            return neohookean_ogden_hex8_energy_element_geometry_soa<scalar_t, VECTOR_SIZE>(nelements, jacobian_adjugate, jacobian_determinant, lmbda, mu, u_streams, values);
        case 100008:
            return neohookean_ogden_proteus_hex8_energy_element_geometry_soa<scalar_t, VECTOR_SIZE>(nelements, jacobian_adjugate, jacobian_determinant, lmbda, mu, u_streams, values);
        default:
            return SFEM_FAILURE;
    }
}

template <typename scalar_t, int VECTOR_SIZE = 16, typename elem_type_t>
static SFEM_INLINE int neohookean_ogden_gradient_2d_element_soa(
        const elem_type_t element_type,
        const ptrdiff_t nelements,
        const scalar_t *const *const SFEM_RESTRICT coords,
        const scalar_t lmbda,
        const scalar_t mu,
        const scalar_t *const *const SFEM_RESTRICT u_streams,
        scalar_t *const *const SFEM_RESTRICT out_streams
) {
    switch ((int)element_type) {
        case 3:
            return neohookean_ogden_tri3_gradient_element_soa<scalar_t, VECTOR_SIZE>(nelements, coords, lmbda, mu, u_streams, out_streams);
        case 40:
            return neohookean_ogden_quad4_gradient_element_soa<scalar_t, VECTOR_SIZE>(nelements, coords, lmbda, mu, u_streams, out_streams);
        case 400000:
            return neohookean_ogden_proteus_quad4_gradient_element_soa<scalar_t, VECTOR_SIZE>(nelements, coords, lmbda, mu, u_streams, out_streams);
        default:
            return SFEM_FAILURE;
    }
}

template <typename scalar_t, int VECTOR_SIZE = 16, typename elem_type_t>
static SFEM_INLINE int neohookean_ogden_gradient_3d_element_soa(
        const elem_type_t element_type,
        const ptrdiff_t nelements,
        const scalar_t *const *const SFEM_RESTRICT coords,
        const scalar_t lmbda,
        const scalar_t mu,
        const scalar_t *const *const SFEM_RESTRICT u_streams,
        scalar_t *const *const SFEM_RESTRICT out_streams
) {
    switch ((int)element_type) {
        case 4:
            return neohookean_ogden_tet4_gradient_element_soa<scalar_t, VECTOR_SIZE>(nelements, coords, lmbda, mu, u_streams, out_streams);
        case 10:
            return neohookean_ogden_tet10_gradient_element_soa<scalar_t, VECTOR_SIZE>(nelements, coords, lmbda, mu, u_streams, out_streams);
        case 8:
            return neohookean_ogden_hex8_gradient_element_soa<scalar_t, VECTOR_SIZE>(nelements, coords, lmbda, mu, u_streams, out_streams);
        case 100008:
            return neohookean_ogden_proteus_hex8_gradient_element_soa<scalar_t, VECTOR_SIZE>(nelements, coords, lmbda, mu, u_streams, out_streams);
        default:
            return SFEM_FAILURE;
    }
}

template <typename scalar_t, int VECTOR_SIZE = 16, typename elem_type_t>
static SFEM_INLINE int neohookean_ogden_gradient_2d_element_coords_soa(
        const elem_type_t element_type,
        const ptrdiff_t nelements,
        const scalar_t *const *const SFEM_RESTRICT coords,
        const scalar_t lmbda,
        const scalar_t mu,
        const scalar_t *const *const SFEM_RESTRICT u_streams,
        scalar_t *const *const SFEM_RESTRICT out_streams
) {
    switch ((int)element_type) {
        case 3:
            return neohookean_ogden_tri3_gradient_element_coords_soa<scalar_t, VECTOR_SIZE>(nelements, coords, lmbda, mu, u_streams, out_streams);
        case 40:
            return neohookean_ogden_quad4_gradient_element_coords_soa<scalar_t, VECTOR_SIZE>(nelements, coords, lmbda, mu, u_streams, out_streams);
        case 400000:
            return neohookean_ogden_proteus_quad4_gradient_element_coords_soa<scalar_t, VECTOR_SIZE>(nelements, coords, lmbda, mu, u_streams, out_streams);
        default:
            return SFEM_FAILURE;
    }
}

template <typename scalar_t, int VECTOR_SIZE = 16, typename elem_type_t>
static SFEM_INLINE int neohookean_ogden_gradient_3d_element_coords_soa(
        const elem_type_t element_type,
        const ptrdiff_t nelements,
        const scalar_t *const *const SFEM_RESTRICT coords,
        const scalar_t lmbda,
        const scalar_t mu,
        const scalar_t *const *const SFEM_RESTRICT u_streams,
        scalar_t *const *const SFEM_RESTRICT out_streams
) {
    switch ((int)element_type) {
        case 4:
            return neohookean_ogden_tet4_gradient_element_coords_soa<scalar_t, VECTOR_SIZE>(nelements, coords, lmbda, mu, u_streams, out_streams);
        case 10:
            return neohookean_ogden_tet10_gradient_element_coords_soa<scalar_t, VECTOR_SIZE>(nelements, coords, lmbda, mu, u_streams, out_streams);
        case 8:
            return neohookean_ogden_hex8_gradient_element_coords_soa<scalar_t, VECTOR_SIZE>(nelements, coords, lmbda, mu, u_streams, out_streams);
        case 100008:
            return neohookean_ogden_proteus_hex8_gradient_element_coords_soa<scalar_t, VECTOR_SIZE>(nelements, coords, lmbda, mu, u_streams, out_streams);
        default:
            return SFEM_FAILURE;
    }
}

template <typename scalar_t, int VECTOR_SIZE = 16, typename elem_type_t>
static SFEM_INLINE int neohookean_ogden_gradient_2d_element_geometry_soa(
        const elem_type_t element_type,
        const ptrdiff_t nelements,
        const scalar_t *const *const SFEM_RESTRICT jacobian_adjugate,
        const scalar_t *const SFEM_RESTRICT jacobian_determinant,
        const scalar_t lmbda,
        const scalar_t mu,
        const scalar_t *const *const SFEM_RESTRICT u_streams,
        scalar_t *const *const SFEM_RESTRICT out_streams
) {
    switch ((int)element_type) {
        case 3:
            return neohookean_ogden_tri3_gradient_element_geometry_soa<scalar_t, VECTOR_SIZE>(nelements, jacobian_adjugate, jacobian_determinant, lmbda, mu, u_streams, out_streams);
        case 40:
            return neohookean_ogden_quad4_gradient_element_geometry_soa<scalar_t, VECTOR_SIZE>(nelements, jacobian_adjugate, jacobian_determinant, lmbda, mu, u_streams, out_streams);
        case 400000:
            return neohookean_ogden_proteus_quad4_gradient_element_geometry_soa<scalar_t, VECTOR_SIZE>(nelements, jacobian_adjugate, jacobian_determinant, lmbda, mu, u_streams, out_streams);
        default:
            return SFEM_FAILURE;
    }
}

template <typename scalar_t, int VECTOR_SIZE = 16, typename elem_type_t>
static SFEM_INLINE int neohookean_ogden_gradient_3d_element_geometry_soa(
        const elem_type_t element_type,
        const ptrdiff_t nelements,
        const scalar_t *const *const SFEM_RESTRICT jacobian_adjugate,
        const scalar_t *const SFEM_RESTRICT jacobian_determinant,
        const scalar_t lmbda,
        const scalar_t mu,
        const scalar_t *const *const SFEM_RESTRICT u_streams,
        scalar_t *const *const SFEM_RESTRICT out_streams
) {
    switch ((int)element_type) {
        case 4:
            return neohookean_ogden_tet4_gradient_element_geometry_soa<scalar_t, VECTOR_SIZE>(nelements, jacobian_adjugate, jacobian_determinant, lmbda, mu, u_streams, out_streams);
        case 10:
            return neohookean_ogden_tet10_gradient_element_geometry_soa<scalar_t, VECTOR_SIZE>(nelements, jacobian_adjugate, jacobian_determinant, lmbda, mu, u_streams, out_streams);
        case 8:
            return neohookean_ogden_hex8_gradient_element_geometry_soa<scalar_t, VECTOR_SIZE>(nelements, jacobian_adjugate, jacobian_determinant, lmbda, mu, u_streams, out_streams);
        case 100008:
            return neohookean_ogden_proteus_hex8_gradient_element_geometry_soa<scalar_t, VECTOR_SIZE>(nelements, jacobian_adjugate, jacobian_determinant, lmbda, mu, u_streams, out_streams);
        default:
            return SFEM_FAILURE;
    }
}

template <typename scalar_t, int VECTOR_SIZE = 16, typename elem_type_t>
static SFEM_INLINE int neohookean_ogden_hessian_2d_element_soa(
        const elem_type_t element_type,
        const ptrdiff_t nelements,
        const scalar_t *const *const SFEM_RESTRICT coords,
        const scalar_t lmbda,
        const scalar_t mu,
        const scalar_t *const *const SFEM_RESTRICT u_streams,
        scalar_t *const *const SFEM_RESTRICT matrix_streams
) {
    switch ((int)element_type) {
        case 3:
            return neohookean_ogden_tri3_hessian_element_soa<scalar_t, VECTOR_SIZE>(nelements, coords, lmbda, mu, u_streams, matrix_streams);
        case 40:
            return neohookean_ogden_quad4_hessian_element_soa<scalar_t, VECTOR_SIZE>(nelements, coords, lmbda, mu, u_streams, matrix_streams);
        case 400000:
            return neohookean_ogden_proteus_quad4_hessian_element_soa<scalar_t, VECTOR_SIZE>(nelements, coords, lmbda, mu, u_streams, matrix_streams);
        default:
            return SFEM_FAILURE;
    }
}

template <typename scalar_t, int VECTOR_SIZE = 16, typename elem_type_t>
static SFEM_INLINE int neohookean_ogden_hessian_3d_element_soa(
        const elem_type_t element_type,
        const ptrdiff_t nelements,
        const scalar_t *const *const SFEM_RESTRICT coords,
        const scalar_t lmbda,
        const scalar_t mu,
        const scalar_t *const *const SFEM_RESTRICT u_streams,
        scalar_t *const *const SFEM_RESTRICT matrix_streams
) {
    switch ((int)element_type) {
        case 4:
            return neohookean_ogden_tet4_hessian_element_soa<scalar_t, VECTOR_SIZE>(nelements, coords, lmbda, mu, u_streams, matrix_streams);
        case 10:
            return neohookean_ogden_tet10_hessian_element_soa<scalar_t, VECTOR_SIZE>(nelements, coords, lmbda, mu, u_streams, matrix_streams);
        case 8:
            return neohookean_ogden_hex8_hessian_element_soa<scalar_t, VECTOR_SIZE>(nelements, coords, lmbda, mu, u_streams, matrix_streams);
        case 100008:
            return neohookean_ogden_proteus_hex8_hessian_element_soa<scalar_t, VECTOR_SIZE>(nelements, coords, lmbda, mu, u_streams, matrix_streams);
        default:
            return SFEM_FAILURE;
    }
}

template <typename scalar_t, int VECTOR_SIZE = 16, typename elem_type_t>
static SFEM_INLINE int neohookean_ogden_hessian_2d_element_coords_soa(
        const elem_type_t element_type,
        const ptrdiff_t nelements,
        const scalar_t *const *const SFEM_RESTRICT coords,
        const scalar_t lmbda,
        const scalar_t mu,
        const scalar_t *const *const SFEM_RESTRICT u_streams,
        scalar_t *const *const SFEM_RESTRICT matrix_streams
) {
    switch ((int)element_type) {
        case 3:
            return neohookean_ogden_tri3_hessian_element_coords_soa<scalar_t, VECTOR_SIZE>(nelements, coords, lmbda, mu, u_streams, matrix_streams);
        case 40:
            return neohookean_ogden_quad4_hessian_element_coords_soa<scalar_t, VECTOR_SIZE>(nelements, coords, lmbda, mu, u_streams, matrix_streams);
        case 400000:
            return neohookean_ogden_proteus_quad4_hessian_element_coords_soa<scalar_t, VECTOR_SIZE>(nelements, coords, lmbda, mu, u_streams, matrix_streams);
        default:
            return SFEM_FAILURE;
    }
}

template <typename scalar_t, int VECTOR_SIZE = 16, typename elem_type_t>
static SFEM_INLINE int neohookean_ogden_hessian_3d_element_coords_soa(
        const elem_type_t element_type,
        const ptrdiff_t nelements,
        const scalar_t *const *const SFEM_RESTRICT coords,
        const scalar_t lmbda,
        const scalar_t mu,
        const scalar_t *const *const SFEM_RESTRICT u_streams,
        scalar_t *const *const SFEM_RESTRICT matrix_streams
) {
    switch ((int)element_type) {
        case 4:
            return neohookean_ogden_tet4_hessian_element_coords_soa<scalar_t, VECTOR_SIZE>(nelements, coords, lmbda, mu, u_streams, matrix_streams);
        case 10:
            return neohookean_ogden_tet10_hessian_element_coords_soa<scalar_t, VECTOR_SIZE>(nelements, coords, lmbda, mu, u_streams, matrix_streams);
        case 8:
            return neohookean_ogden_hex8_hessian_element_coords_soa<scalar_t, VECTOR_SIZE>(nelements, coords, lmbda, mu, u_streams, matrix_streams);
        case 100008:
            return neohookean_ogden_proteus_hex8_hessian_element_coords_soa<scalar_t, VECTOR_SIZE>(nelements, coords, lmbda, mu, u_streams, matrix_streams);
        default:
            return SFEM_FAILURE;
    }
}

template <typename scalar_t, int VECTOR_SIZE = 16, typename elem_type_t>
static SFEM_INLINE int neohookean_ogden_hessian_2d_element_geometry_soa(
        const elem_type_t element_type,
        const ptrdiff_t nelements,
        const scalar_t *const *const SFEM_RESTRICT jacobian_adjugate,
        const scalar_t *const SFEM_RESTRICT jacobian_determinant,
        const scalar_t lmbda,
        const scalar_t mu,
        const scalar_t *const *const SFEM_RESTRICT u_streams,
        scalar_t *const *const SFEM_RESTRICT matrix_streams
) {
    switch ((int)element_type) {
        case 3:
            return neohookean_ogden_tri3_hessian_element_geometry_soa<scalar_t, VECTOR_SIZE>(nelements, jacobian_adjugate, jacobian_determinant, lmbda, mu, u_streams, matrix_streams);
        case 40:
            return neohookean_ogden_quad4_hessian_element_geometry_soa<scalar_t, VECTOR_SIZE>(nelements, jacobian_adjugate, jacobian_determinant, lmbda, mu, u_streams, matrix_streams);
        case 400000:
            return neohookean_ogden_proteus_quad4_hessian_element_geometry_soa<scalar_t, VECTOR_SIZE>(nelements, jacobian_adjugate, jacobian_determinant, lmbda, mu, u_streams, matrix_streams);
        default:
            return SFEM_FAILURE;
    }
}

template <typename scalar_t, int VECTOR_SIZE = 16, typename elem_type_t>
static SFEM_INLINE int neohookean_ogden_hessian_3d_element_geometry_soa(
        const elem_type_t element_type,
        const ptrdiff_t nelements,
        const scalar_t *const *const SFEM_RESTRICT jacobian_adjugate,
        const scalar_t *const SFEM_RESTRICT jacobian_determinant,
        const scalar_t lmbda,
        const scalar_t mu,
        const scalar_t *const *const SFEM_RESTRICT u_streams,
        scalar_t *const *const SFEM_RESTRICT matrix_streams
) {
    switch ((int)element_type) {
        case 4:
            return neohookean_ogden_tet4_hessian_element_geometry_soa<scalar_t, VECTOR_SIZE>(nelements, jacobian_adjugate, jacobian_determinant, lmbda, mu, u_streams, matrix_streams);
        case 10:
            return neohookean_ogden_tet10_hessian_element_geometry_soa<scalar_t, VECTOR_SIZE>(nelements, jacobian_adjugate, jacobian_determinant, lmbda, mu, u_streams, matrix_streams);
        case 8:
            return neohookean_ogden_hex8_hessian_element_geometry_soa<scalar_t, VECTOR_SIZE>(nelements, jacobian_adjugate, jacobian_determinant, lmbda, mu, u_streams, matrix_streams);
        case 100008:
            return neohookean_ogden_proteus_hex8_hessian_element_geometry_soa<scalar_t, VECTOR_SIZE>(nelements, jacobian_adjugate, jacobian_determinant, lmbda, mu, u_streams, matrix_streams);
        default:
            return SFEM_FAILURE;
    }
}

} // namespace codegen
} // namespace sfem
