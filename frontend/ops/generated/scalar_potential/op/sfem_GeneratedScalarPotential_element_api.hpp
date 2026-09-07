#pragma once

#include <cstddef>

#ifndef SFEM_SUCCESS
#define SFEM_SUCCESS 0
#endif
#ifndef SFEM_FAILURE
#define SFEM_FAILURE 1
#endif

#include "../d2/proteus_quad4/scalar_potential_proteus_quad4_element.hpp"
#include "../d2/quad4/scalar_potential_quad4_element.hpp"
#include "../d2/tri3/scalar_potential_tri3_element.hpp"
#include "../d3/hex8/scalar_potential_hex8_element.hpp"
#include "../d3/proteus_hex8/scalar_potential_proteus_hex8_element.hpp"
#include "../d3/tet4/scalar_potential_tet4_element.hpp"

namespace sfem {
namespace codegen {

template <typename scalar_t, int VECTOR_SIZE = 16, typename elem_type_t>
static SFEM_INLINE int scalar_potential_energy_2d_element_soa(
        const elem_type_t element_type,
        const ptrdiff_t nelements,
        const scalar_t *const *const SFEM_RESTRICT coords,
        const scalar_t kappa,
        const scalar_t *const *const SFEM_RESTRICT u_streams,
        scalar_t *const SFEM_RESTRICT values
) {
    switch ((int)element_type) {
        case 3:
            return scalar_potential_tri3_energy_element_soa<scalar_t, VECTOR_SIZE>(nelements, coords, kappa, u_streams, values);
        case 40:
            return scalar_potential_quad4_energy_element_soa<scalar_t, VECTOR_SIZE>(nelements, coords, kappa, u_streams, values);
        case 400000:
            return scalar_potential_proteus_quad4_energy_element_soa<scalar_t, VECTOR_SIZE>(nelements, coords, kappa, u_streams, values);
        default:
            return SFEM_FAILURE;
    }
}

template <typename scalar_t, int VECTOR_SIZE = 16, typename elem_type_t>
static SFEM_INLINE int scalar_potential_energy_3d_element_soa(
        const elem_type_t element_type,
        const ptrdiff_t nelements,
        const scalar_t *const *const SFEM_RESTRICT coords,
        const scalar_t kappa,
        const scalar_t *const *const SFEM_RESTRICT u_streams,
        scalar_t *const SFEM_RESTRICT values
) {
    switch ((int)element_type) {
        case 4:
            return scalar_potential_tet4_energy_element_soa<scalar_t, VECTOR_SIZE>(nelements, coords, kappa, u_streams, values);
        case 8:
            return scalar_potential_hex8_energy_element_soa<scalar_t, VECTOR_SIZE>(nelements, coords, kappa, u_streams, values);
        case 100008:
            return scalar_potential_proteus_hex8_energy_element_soa<scalar_t, VECTOR_SIZE>(nelements, coords, kappa, u_streams, values);
        default:
            return SFEM_FAILURE;
    }
}

template <typename scalar_t, int VECTOR_SIZE = 16, typename elem_type_t>
static SFEM_INLINE int scalar_potential_energy_2d_element_coords_soa(
        const elem_type_t element_type,
        const ptrdiff_t nelements,
        const scalar_t *const *const SFEM_RESTRICT coords,
        const scalar_t kappa,
        const scalar_t *const *const SFEM_RESTRICT u_streams,
        scalar_t *const SFEM_RESTRICT values
) {
    switch ((int)element_type) {
        case 3:
            return scalar_potential_tri3_energy_element_coords_soa<scalar_t, VECTOR_SIZE>(nelements, coords, kappa, u_streams, values);
        case 40:
            return scalar_potential_quad4_energy_element_coords_soa<scalar_t, VECTOR_SIZE>(nelements, coords, kappa, u_streams, values);
        case 400000:
            return scalar_potential_proteus_quad4_energy_element_coords_soa<scalar_t, VECTOR_SIZE>(nelements, coords, kappa, u_streams, values);
        default:
            return SFEM_FAILURE;
    }
}

template <typename scalar_t, int VECTOR_SIZE = 16, typename elem_type_t>
static SFEM_INLINE int scalar_potential_energy_3d_element_coords_soa(
        const elem_type_t element_type,
        const ptrdiff_t nelements,
        const scalar_t *const *const SFEM_RESTRICT coords,
        const scalar_t kappa,
        const scalar_t *const *const SFEM_RESTRICT u_streams,
        scalar_t *const SFEM_RESTRICT values
) {
    switch ((int)element_type) {
        case 4:
            return scalar_potential_tet4_energy_element_coords_soa<scalar_t, VECTOR_SIZE>(nelements, coords, kappa, u_streams, values);
        case 8:
            return scalar_potential_hex8_energy_element_coords_soa<scalar_t, VECTOR_SIZE>(nelements, coords, kappa, u_streams, values);
        case 100008:
            return scalar_potential_proteus_hex8_energy_element_coords_soa<scalar_t, VECTOR_SIZE>(nelements, coords, kappa, u_streams, values);
        default:
            return SFEM_FAILURE;
    }
}

template <typename scalar_t, int VECTOR_SIZE = 16, typename elem_type_t>
static SFEM_INLINE int scalar_potential_energy_2d_element_geometry_soa(
        const elem_type_t element_type,
        const ptrdiff_t nelements,
        const scalar_t *const *const SFEM_RESTRICT jacobian_adjugate,
        const scalar_t *const SFEM_RESTRICT jacobian_determinant,
        const scalar_t kappa,
        const scalar_t *const *const SFEM_RESTRICT u_streams,
        scalar_t *const SFEM_RESTRICT values
) {
    switch ((int)element_type) {
        case 3:
            return scalar_potential_tri3_energy_element_geometry_soa<scalar_t, VECTOR_SIZE>(nelements, jacobian_adjugate, jacobian_determinant, kappa, u_streams, values);
        case 40:
            return scalar_potential_quad4_energy_element_geometry_soa<scalar_t, VECTOR_SIZE>(nelements, jacobian_adjugate, jacobian_determinant, kappa, u_streams, values);
        case 400000:
            return scalar_potential_proteus_quad4_energy_element_geometry_soa<scalar_t, VECTOR_SIZE>(nelements, jacobian_adjugate, jacobian_determinant, kappa, u_streams, values);
        default:
            return SFEM_FAILURE;
    }
}

template <typename scalar_t, int VECTOR_SIZE = 16, typename elem_type_t>
static SFEM_INLINE int scalar_potential_energy_3d_element_geometry_soa(
        const elem_type_t element_type,
        const ptrdiff_t nelements,
        const scalar_t *const *const SFEM_RESTRICT jacobian_adjugate,
        const scalar_t *const SFEM_RESTRICT jacobian_determinant,
        const scalar_t kappa,
        const scalar_t *const *const SFEM_RESTRICT u_streams,
        scalar_t *const SFEM_RESTRICT values
) {
    switch ((int)element_type) {
        case 4:
            return scalar_potential_tet4_energy_element_geometry_soa<scalar_t, VECTOR_SIZE>(nelements, jacobian_adjugate, jacobian_determinant, kappa, u_streams, values);
        case 8:
            return scalar_potential_hex8_energy_element_geometry_soa<scalar_t, VECTOR_SIZE>(nelements, jacobian_adjugate, jacobian_determinant, kappa, u_streams, values);
        case 100008:
            return scalar_potential_proteus_hex8_energy_element_geometry_soa<scalar_t, VECTOR_SIZE>(nelements, jacobian_adjugate, jacobian_determinant, kappa, u_streams, values);
        default:
            return SFEM_FAILURE;
    }
}

template <typename scalar_t, int VECTOR_SIZE = 16, typename elem_type_t>
static SFEM_INLINE int scalar_potential_gradient_2d_element_soa(
        const elem_type_t element_type,
        const ptrdiff_t nelements,
        const scalar_t *const *const SFEM_RESTRICT coords,
        const scalar_t kappa,
        const scalar_t *const *const SFEM_RESTRICT u_streams,
        scalar_t *const *const SFEM_RESTRICT out_streams
) {
    switch ((int)element_type) {
        case 3:
            return scalar_potential_tri3_gradient_element_soa<scalar_t, VECTOR_SIZE>(nelements, coords, kappa, u_streams, out_streams);
        case 40:
            return scalar_potential_quad4_gradient_element_soa<scalar_t, VECTOR_SIZE>(nelements, coords, kappa, u_streams, out_streams);
        case 400000:
            return scalar_potential_proteus_quad4_gradient_element_soa<scalar_t, VECTOR_SIZE>(nelements, coords, kappa, u_streams, out_streams);
        default:
            return SFEM_FAILURE;
    }
}

template <typename scalar_t, int VECTOR_SIZE = 16, typename elem_type_t>
static SFEM_INLINE int scalar_potential_gradient_3d_element_soa(
        const elem_type_t element_type,
        const ptrdiff_t nelements,
        const scalar_t *const *const SFEM_RESTRICT coords,
        const scalar_t kappa,
        const scalar_t *const *const SFEM_RESTRICT u_streams,
        scalar_t *const *const SFEM_RESTRICT out_streams
) {
    switch ((int)element_type) {
        case 4:
            return scalar_potential_tet4_gradient_element_soa<scalar_t, VECTOR_SIZE>(nelements, coords, kappa, u_streams, out_streams);
        case 8:
            return scalar_potential_hex8_gradient_element_soa<scalar_t, VECTOR_SIZE>(nelements, coords, kappa, u_streams, out_streams);
        case 100008:
            return scalar_potential_proteus_hex8_gradient_element_soa<scalar_t, VECTOR_SIZE>(nelements, coords, kappa, u_streams, out_streams);
        default:
            return SFEM_FAILURE;
    }
}

template <typename scalar_t, int VECTOR_SIZE = 16, typename elem_type_t>
static SFEM_INLINE int scalar_potential_gradient_2d_element_coords_soa(
        const elem_type_t element_type,
        const ptrdiff_t nelements,
        const scalar_t *const *const SFEM_RESTRICT coords,
        const scalar_t kappa,
        const scalar_t *const *const SFEM_RESTRICT u_streams,
        scalar_t *const *const SFEM_RESTRICT out_streams
) {
    switch ((int)element_type) {
        case 3:
            return scalar_potential_tri3_gradient_element_coords_soa<scalar_t, VECTOR_SIZE>(nelements, coords, kappa, u_streams, out_streams);
        case 40:
            return scalar_potential_quad4_gradient_element_coords_soa<scalar_t, VECTOR_SIZE>(nelements, coords, kappa, u_streams, out_streams);
        case 400000:
            return scalar_potential_proteus_quad4_gradient_element_coords_soa<scalar_t, VECTOR_SIZE>(nelements, coords, kappa, u_streams, out_streams);
        default:
            return SFEM_FAILURE;
    }
}

template <typename scalar_t, int VECTOR_SIZE = 16, typename elem_type_t>
static SFEM_INLINE int scalar_potential_gradient_3d_element_coords_soa(
        const elem_type_t element_type,
        const ptrdiff_t nelements,
        const scalar_t *const *const SFEM_RESTRICT coords,
        const scalar_t kappa,
        const scalar_t *const *const SFEM_RESTRICT u_streams,
        scalar_t *const *const SFEM_RESTRICT out_streams
) {
    switch ((int)element_type) {
        case 4:
            return scalar_potential_tet4_gradient_element_coords_soa<scalar_t, VECTOR_SIZE>(nelements, coords, kappa, u_streams, out_streams);
        case 8:
            return scalar_potential_hex8_gradient_element_coords_soa<scalar_t, VECTOR_SIZE>(nelements, coords, kappa, u_streams, out_streams);
        case 100008:
            return scalar_potential_proteus_hex8_gradient_element_coords_soa<scalar_t, VECTOR_SIZE>(nelements, coords, kappa, u_streams, out_streams);
        default:
            return SFEM_FAILURE;
    }
}

template <typename scalar_t, int VECTOR_SIZE = 16, typename elem_type_t>
static SFEM_INLINE int scalar_potential_gradient_2d_element_geometry_soa(
        const elem_type_t element_type,
        const ptrdiff_t nelements,
        const scalar_t *const *const SFEM_RESTRICT jacobian_adjugate,
        const scalar_t *const SFEM_RESTRICT jacobian_determinant,
        const scalar_t kappa,
        const scalar_t *const *const SFEM_RESTRICT u_streams,
        scalar_t *const *const SFEM_RESTRICT out_streams
) {
    switch ((int)element_type) {
        case 3:
            return scalar_potential_tri3_gradient_element_geometry_soa<scalar_t, VECTOR_SIZE>(nelements, jacobian_adjugate, jacobian_determinant, kappa, u_streams, out_streams);
        case 40:
            return scalar_potential_quad4_gradient_element_geometry_soa<scalar_t, VECTOR_SIZE>(nelements, jacobian_adjugate, jacobian_determinant, kappa, u_streams, out_streams);
        case 400000:
            return scalar_potential_proteus_quad4_gradient_element_geometry_soa<scalar_t, VECTOR_SIZE>(nelements, jacobian_adjugate, jacobian_determinant, kappa, u_streams, out_streams);
        default:
            return SFEM_FAILURE;
    }
}

template <typename scalar_t, int VECTOR_SIZE = 16, typename elem_type_t>
static SFEM_INLINE int scalar_potential_gradient_3d_element_geometry_soa(
        const elem_type_t element_type,
        const ptrdiff_t nelements,
        const scalar_t *const *const SFEM_RESTRICT jacobian_adjugate,
        const scalar_t *const SFEM_RESTRICT jacobian_determinant,
        const scalar_t kappa,
        const scalar_t *const *const SFEM_RESTRICT u_streams,
        scalar_t *const *const SFEM_RESTRICT out_streams
) {
    switch ((int)element_type) {
        case 4:
            return scalar_potential_tet4_gradient_element_geometry_soa<scalar_t, VECTOR_SIZE>(nelements, jacobian_adjugate, jacobian_determinant, kappa, u_streams, out_streams);
        case 8:
            return scalar_potential_hex8_gradient_element_geometry_soa<scalar_t, VECTOR_SIZE>(nelements, jacobian_adjugate, jacobian_determinant, kappa, u_streams, out_streams);
        case 100008:
            return scalar_potential_proteus_hex8_gradient_element_geometry_soa<scalar_t, VECTOR_SIZE>(nelements, jacobian_adjugate, jacobian_determinant, kappa, u_streams, out_streams);
        default:
            return SFEM_FAILURE;
    }
}

template <typename scalar_t, int VECTOR_SIZE = 16, typename elem_type_t>
static SFEM_INLINE int scalar_potential_hessian_2d_element_soa(
        const elem_type_t element_type,
        const ptrdiff_t nelements,
        const scalar_t *const *const SFEM_RESTRICT coords,
        const scalar_t kappa,
        scalar_t *const *const SFEM_RESTRICT matrix_streams
) {
    switch ((int)element_type) {
        case 3:
            return scalar_potential_tri3_hessian_element_soa<scalar_t, VECTOR_SIZE>(nelements, coords, kappa, matrix_streams);
        case 40:
            return scalar_potential_quad4_hessian_element_soa<scalar_t, VECTOR_SIZE>(nelements, coords, kappa, matrix_streams);
        case 400000:
            return scalar_potential_proteus_quad4_hessian_element_soa<scalar_t, VECTOR_SIZE>(nelements, coords, kappa, matrix_streams);
        default:
            return SFEM_FAILURE;
    }
}

template <typename scalar_t, int VECTOR_SIZE = 16, typename elem_type_t>
static SFEM_INLINE int scalar_potential_hessian_3d_element_soa(
        const elem_type_t element_type,
        const ptrdiff_t nelements,
        const scalar_t *const *const SFEM_RESTRICT coords,
        const scalar_t kappa,
        scalar_t *const *const SFEM_RESTRICT matrix_streams
) {
    switch ((int)element_type) {
        case 4:
            return scalar_potential_tet4_hessian_element_soa<scalar_t, VECTOR_SIZE>(nelements, coords, kappa, matrix_streams);
        case 8:
            return scalar_potential_hex8_hessian_element_soa<scalar_t, VECTOR_SIZE>(nelements, coords, kappa, matrix_streams);
        case 100008:
            return scalar_potential_proteus_hex8_hessian_element_soa<scalar_t, VECTOR_SIZE>(nelements, coords, kappa, matrix_streams);
        default:
            return SFEM_FAILURE;
    }
}

template <typename scalar_t, int VECTOR_SIZE = 16, typename elem_type_t>
static SFEM_INLINE int scalar_potential_hessian_2d_element_coords_soa(
        const elem_type_t element_type,
        const ptrdiff_t nelements,
        const scalar_t *const *const SFEM_RESTRICT coords,
        const scalar_t kappa,
        scalar_t *const *const SFEM_RESTRICT matrix_streams
) {
    switch ((int)element_type) {
        case 3:
            return scalar_potential_tri3_hessian_element_coords_soa<scalar_t, VECTOR_SIZE>(nelements, coords, kappa, matrix_streams);
        case 40:
            return scalar_potential_quad4_hessian_element_coords_soa<scalar_t, VECTOR_SIZE>(nelements, coords, kappa, matrix_streams);
        case 400000:
            return scalar_potential_proteus_quad4_hessian_element_coords_soa<scalar_t, VECTOR_SIZE>(nelements, coords, kappa, matrix_streams);
        default:
            return SFEM_FAILURE;
    }
}

template <typename scalar_t, int VECTOR_SIZE = 16, typename elem_type_t>
static SFEM_INLINE int scalar_potential_hessian_3d_element_coords_soa(
        const elem_type_t element_type,
        const ptrdiff_t nelements,
        const scalar_t *const *const SFEM_RESTRICT coords,
        const scalar_t kappa,
        scalar_t *const *const SFEM_RESTRICT matrix_streams
) {
    switch ((int)element_type) {
        case 4:
            return scalar_potential_tet4_hessian_element_coords_soa<scalar_t, VECTOR_SIZE>(nelements, coords, kappa, matrix_streams);
        case 8:
            return scalar_potential_hex8_hessian_element_coords_soa<scalar_t, VECTOR_SIZE>(nelements, coords, kappa, matrix_streams);
        case 100008:
            return scalar_potential_proteus_hex8_hessian_element_coords_soa<scalar_t, VECTOR_SIZE>(nelements, coords, kappa, matrix_streams);
        default:
            return SFEM_FAILURE;
    }
}

template <typename scalar_t, int VECTOR_SIZE = 16, typename elem_type_t>
static SFEM_INLINE int scalar_potential_hessian_2d_element_geometry_soa(
        const elem_type_t element_type,
        const ptrdiff_t nelements,
        const scalar_t *const *const SFEM_RESTRICT jacobian_adjugate,
        const scalar_t *const SFEM_RESTRICT jacobian_determinant,
        const scalar_t kappa,
        scalar_t *const *const SFEM_RESTRICT matrix_streams
) {
    switch ((int)element_type) {
        case 3:
            return scalar_potential_tri3_hessian_element_geometry_soa<scalar_t, VECTOR_SIZE>(nelements, jacobian_adjugate, jacobian_determinant, kappa, matrix_streams);
        case 40:
            return scalar_potential_quad4_hessian_element_geometry_soa<scalar_t, VECTOR_SIZE>(nelements, jacobian_adjugate, jacobian_determinant, kappa, matrix_streams);
        case 400000:
            return scalar_potential_proteus_quad4_hessian_element_geometry_soa<scalar_t, VECTOR_SIZE>(nelements, jacobian_adjugate, jacobian_determinant, kappa, matrix_streams);
        default:
            return SFEM_FAILURE;
    }
}

template <typename scalar_t, int VECTOR_SIZE = 16, typename elem_type_t>
static SFEM_INLINE int scalar_potential_hessian_3d_element_geometry_soa(
        const elem_type_t element_type,
        const ptrdiff_t nelements,
        const scalar_t *const *const SFEM_RESTRICT jacobian_adjugate,
        const scalar_t *const SFEM_RESTRICT jacobian_determinant,
        const scalar_t kappa,
        scalar_t *const *const SFEM_RESTRICT matrix_streams
) {
    switch ((int)element_type) {
        case 4:
            return scalar_potential_tet4_hessian_element_geometry_soa<scalar_t, VECTOR_SIZE>(nelements, jacobian_adjugate, jacobian_determinant, kappa, matrix_streams);
        case 8:
            return scalar_potential_hex8_hessian_element_geometry_soa<scalar_t, VECTOR_SIZE>(nelements, jacobian_adjugate, jacobian_determinant, kappa, matrix_streams);
        case 100008:
            return scalar_potential_proteus_hex8_hessian_element_geometry_soa<scalar_t, VECTOR_SIZE>(nelements, jacobian_adjugate, jacobian_determinant, kappa, matrix_streams);
        default:
            return SFEM_FAILURE;
    }
}

} // namespace codegen
} // namespace sfem
