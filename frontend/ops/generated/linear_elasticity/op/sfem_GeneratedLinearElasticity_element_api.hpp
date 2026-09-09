#pragma once

#include <cstddef>

#ifndef SFEM_SUCCESS
#define SFEM_SUCCESS 0
#endif
#ifndef SFEM_FAILURE
#define SFEM_FAILURE 1
#endif

#include "../d2/proteus_quad4/linear_elasticity_proteus_quad4_element.hpp"
#include "../d2/quad4/linear_elasticity_quad4_element.hpp"
#include "../d2/tri3/linear_elasticity_tri3_element.hpp"
#include "../d3/hex8/linear_elasticity_hex8_element.hpp"
#include "../d3/proteus_hex8/linear_elasticity_proteus_hex8_element.hpp"
#include "../d3/tet10/linear_elasticity_tet10_element.hpp"
#include "../d3/tet4/linear_elasticity_tet4_element.hpp"

namespace sfem {
namespace codegen {

template <typename s_t, int VS = 16, typename elem_type_t>
static SFEM_INLINE int linear_elasticity_energy_2d_esoa(
    const elem_type_t element_type,
        const ptrdiff_t nelements,
        const s_t *const *const RSTR coords,
        const s_t lmbda,
        const s_t mu,
        const s_t *const *const RSTR u_streams,
        s_t *const RSTR values
) {
  switch ((int)element_type) {
    case 3:
      return linear_elasticity_tri3_energy_esoa<s_t, VS>(nelements, coords, lmbda, mu, u_streams, values);
    case 40:
      return linear_elasticity_quad4_energy_esoa<s_t, VS>(nelements, coords, lmbda, mu, u_streams, values);
    case 400000:
      return linear_elasticity_proteus_quad4_energy_esoa<s_t, VS>(nelements, coords, lmbda, mu, u_streams, values);
    default:
      return SFEM_FAILURE;
  }
}

template <typename s_t, int VS = 16, typename elem_type_t>
static SFEM_INLINE int linear_elasticity_energy_3d_esoa(
    const elem_type_t element_type,
        const ptrdiff_t nelements,
        const s_t *const *const RSTR coords,
        const s_t lmbda,
        const s_t mu,
        const s_t *const *const RSTR u_streams,
        s_t *const RSTR values
) {
  switch ((int)element_type) {
    case 4:
      return linear_elasticity_tet4_energy_esoa<s_t, VS>(nelements, coords, lmbda, mu, u_streams, values);
    case 10:
      return linear_elasticity_tet10_energy_esoa<s_t, VS>(nelements, coords, lmbda, mu, u_streams, values);
    case 8:
      return linear_elasticity_hex8_energy_esoa<s_t, VS>(nelements, coords, lmbda, mu, u_streams, values);
    case 100008:
      return linear_elasticity_proteus_hex8_energy_esoa<s_t, VS>(nelements, coords, lmbda, mu, u_streams, values);
    default:
      return SFEM_FAILURE;
  }
}

template <typename s_t, int VS = 16, typename elem_type_t>
static SFEM_INLINE int linear_elasticity_energy_2d_ecoords_soa(
    const elem_type_t element_type,
        const ptrdiff_t nelements,
        const s_t *const *const RSTR coords,
        const s_t lmbda,
        const s_t mu,
        const s_t *const *const RSTR u_streams,
        s_t *const RSTR values
) {
  switch ((int)element_type) {
    case 3:
      return linear_elasticity_tri3_energy_ecoords_soa<s_t, VS>(nelements, coords, lmbda, mu, u_streams, values);
    case 40:
      return linear_elasticity_quad4_energy_ecoords_soa<s_t, VS>(nelements, coords, lmbda, mu, u_streams, values);
    case 400000:
      return linear_elasticity_proteus_quad4_energy_ecoords_soa<s_t, VS>(nelements, coords, lmbda, mu, u_streams, values);
    default:
      return SFEM_FAILURE;
  }
}

template <typename s_t, int VS = 16, typename elem_type_t>
static SFEM_INLINE int linear_elasticity_energy_3d_ecoords_soa(
    const elem_type_t element_type,
        const ptrdiff_t nelements,
        const s_t *const *const RSTR coords,
        const s_t lmbda,
        const s_t mu,
        const s_t *const *const RSTR u_streams,
        s_t *const RSTR values
) {
  switch ((int)element_type) {
    case 4:
      return linear_elasticity_tet4_energy_ecoords_soa<s_t, VS>(nelements, coords, lmbda, mu, u_streams, values);
    case 10:
      return linear_elasticity_tet10_energy_ecoords_soa<s_t, VS>(nelements, coords, lmbda, mu, u_streams, values);
    case 8:
      return linear_elasticity_hex8_energy_ecoords_soa<s_t, VS>(nelements, coords, lmbda, mu, u_streams, values);
    case 100008:
      return linear_elasticity_proteus_hex8_energy_ecoords_soa<s_t, VS>(nelements, coords, lmbda, mu, u_streams, values);
    default:
      return SFEM_FAILURE;
  }
}

template <typename s_t, int VS = 16, typename elem_type_t>
static SFEM_INLINE int linear_elasticity_energy_2d_egeometry_soa(
    const elem_type_t element_type,
        const ptrdiff_t nelements,
        const s_t *const *const RSTR adj,
        const s_t *const RSTR det,
        const s_t lmbda,
        const s_t mu,
        const s_t *const *const RSTR u_streams,
        s_t *const RSTR values
) {
  switch ((int)element_type) {
    case 3:
      return linear_elasticity_tri3_energy_egeometry_soa<s_t, VS>(nelements, adj, det, lmbda, mu, u_streams, values);
    case 40:
      return linear_elasticity_quad4_energy_egeometry_soa<s_t, VS>(nelements, adj, det, lmbda, mu, u_streams, values);
    case 400000:
      return linear_elasticity_proteus_quad4_energy_egeometry_soa<s_t, VS>(nelements, adj, det, lmbda, mu, u_streams, values);
    default:
      return SFEM_FAILURE;
  }
}

template <typename s_t, int VS = 16, typename elem_type_t>
static SFEM_INLINE int linear_elasticity_energy_3d_egeometry_soa(
    const elem_type_t element_type,
        const ptrdiff_t nelements,
        const s_t *const *const RSTR adj,
        const s_t *const RSTR det,
        const s_t lmbda,
        const s_t mu,
        const s_t *const *const RSTR u_streams,
        s_t *const RSTR values
) {
  switch ((int)element_type) {
    case 4:
      return linear_elasticity_tet4_energy_egeometry_soa<s_t, VS>(nelements, adj, det, lmbda, mu, u_streams, values);
    case 10:
      return linear_elasticity_tet10_energy_egeometry_soa<s_t, VS>(nelements, adj, det, lmbda, mu, u_streams, values);
    case 8:
      return linear_elasticity_hex8_energy_egeometry_soa<s_t, VS>(nelements, adj, det, lmbda, mu, u_streams, values);
    case 100008:
      return linear_elasticity_proteus_hex8_energy_egeometry_soa<s_t, VS>(nelements, adj, det, lmbda, mu, u_streams, values);
    default:
      return SFEM_FAILURE;
  }
}

template <typename s_t, int VS = 16, typename elem_type_t>
static SFEM_INLINE int linear_elasticity_gradient_2d_esoa(
    const elem_type_t element_type,
        const ptrdiff_t nelements,
        const s_t *const *const RSTR coords,
        const s_t lmbda,
        const s_t mu,
        const s_t *const *const RSTR u_streams,
        s_t *const *const RSTR out_streams
) {
  switch ((int)element_type) {
    case 3:
      return linear_elasticity_tri3_gradient_esoa<s_t, VS>(nelements, coords, lmbda, mu, u_streams, out_streams);
    case 40:
      return linear_elasticity_quad4_gradient_esoa<s_t, VS>(nelements, coords, lmbda, mu, u_streams, out_streams);
    case 400000:
      return linear_elasticity_proteus_quad4_gradient_esoa<s_t, VS>(nelements, coords, lmbda, mu, u_streams, out_streams);
    default:
      return SFEM_FAILURE;
  }
}

template <typename s_t, int VS = 16, typename elem_type_t>
static SFEM_INLINE int linear_elasticity_gradient_3d_esoa(
    const elem_type_t element_type,
        const ptrdiff_t nelements,
        const s_t *const *const RSTR coords,
        const s_t lmbda,
        const s_t mu,
        const s_t *const *const RSTR u_streams,
        s_t *const *const RSTR out_streams
) {
  switch ((int)element_type) {
    case 4:
      return linear_elasticity_tet4_gradient_esoa<s_t, VS>(nelements, coords, lmbda, mu, u_streams, out_streams);
    case 10:
      return linear_elasticity_tet10_gradient_esoa<s_t, VS>(nelements, coords, lmbda, mu, u_streams, out_streams);
    case 8:
      return linear_elasticity_hex8_gradient_esoa<s_t, VS>(nelements, coords, lmbda, mu, u_streams, out_streams);
    case 100008:
      return linear_elasticity_proteus_hex8_gradient_esoa<s_t, VS>(nelements, coords, lmbda, mu, u_streams, out_streams);
    default:
      return SFEM_FAILURE;
  }
}

template <typename s_t, int VS = 16, typename elem_type_t>
static SFEM_INLINE int linear_elasticity_gradient_2d_ecoords_soa(
    const elem_type_t element_type,
        const ptrdiff_t nelements,
        const s_t *const *const RSTR coords,
        const s_t lmbda,
        const s_t mu,
        const s_t *const *const RSTR u_streams,
        s_t *const *const RSTR out_streams
) {
  switch ((int)element_type) {
    case 3:
      return linear_elasticity_tri3_gradient_ecoords_soa<s_t, VS>(nelements, coords, lmbda, mu, u_streams, out_streams);
    case 40:
      return linear_elasticity_quad4_gradient_ecoords_soa<s_t, VS>(nelements, coords, lmbda, mu, u_streams, out_streams);
    case 400000:
      return linear_elasticity_proteus_quad4_gradient_ecoords_soa<s_t, VS>(nelements, coords, lmbda, mu, u_streams, out_streams);
    default:
      return SFEM_FAILURE;
  }
}

template <typename s_t, int VS = 16, typename elem_type_t>
static SFEM_INLINE int linear_elasticity_gradient_3d_ecoords_soa(
    const elem_type_t element_type,
        const ptrdiff_t nelements,
        const s_t *const *const RSTR coords,
        const s_t lmbda,
        const s_t mu,
        const s_t *const *const RSTR u_streams,
        s_t *const *const RSTR out_streams
) {
  switch ((int)element_type) {
    case 4:
      return linear_elasticity_tet4_gradient_ecoords_soa<s_t, VS>(nelements, coords, lmbda, mu, u_streams, out_streams);
    case 10:
      return linear_elasticity_tet10_gradient_ecoords_soa<s_t, VS>(nelements, coords, lmbda, mu, u_streams, out_streams);
    case 8:
      return linear_elasticity_hex8_gradient_ecoords_soa<s_t, VS>(nelements, coords, lmbda, mu, u_streams, out_streams);
    case 100008:
      return linear_elasticity_proteus_hex8_gradient_ecoords_soa<s_t, VS>(nelements, coords, lmbda, mu, u_streams, out_streams);
    default:
      return SFEM_FAILURE;
  }
}

template <typename s_t, int VS = 16, typename elem_type_t>
static SFEM_INLINE int linear_elasticity_gradient_2d_egeometry_soa(
    const elem_type_t element_type,
        const ptrdiff_t nelements,
        const s_t *const *const RSTR adj,
        const s_t *const RSTR det,
        const s_t lmbda,
        const s_t mu,
        const s_t *const *const RSTR u_streams,
        s_t *const *const RSTR out_streams
) {
  switch ((int)element_type) {
    case 3:
      return linear_elasticity_tri3_gradient_egeometry_soa<s_t, VS>(nelements, adj, det, lmbda, mu, u_streams, out_streams);
    case 40:
      return linear_elasticity_quad4_gradient_egeometry_soa<s_t, VS>(nelements, adj, det, lmbda, mu, u_streams, out_streams);
    case 400000:
      return linear_elasticity_proteus_quad4_gradient_egeometry_soa<s_t, VS>(nelements, adj, det, lmbda, mu, u_streams, out_streams);
    default:
      return SFEM_FAILURE;
  }
}

template <typename s_t, int VS = 16, typename elem_type_t>
static SFEM_INLINE int linear_elasticity_gradient_3d_egeometry_soa(
    const elem_type_t element_type,
        const ptrdiff_t nelements,
        const s_t *const *const RSTR adj,
        const s_t *const RSTR det,
        const s_t lmbda,
        const s_t mu,
        const s_t *const *const RSTR u_streams,
        s_t *const *const RSTR out_streams
) {
  switch ((int)element_type) {
    case 4:
      return linear_elasticity_tet4_gradient_egeometry_soa<s_t, VS>(nelements, adj, det, lmbda, mu, u_streams, out_streams);
    case 10:
      return linear_elasticity_tet10_gradient_egeometry_soa<s_t, VS>(nelements, adj, det, lmbda, mu, u_streams, out_streams);
    case 8:
      return linear_elasticity_hex8_gradient_egeometry_soa<s_t, VS>(nelements, adj, det, lmbda, mu, u_streams, out_streams);
    case 100008:
      return linear_elasticity_proteus_hex8_gradient_egeometry_soa<s_t, VS>(nelements, adj, det, lmbda, mu, u_streams, out_streams);
    default:
      return SFEM_FAILURE;
  }
}

template <typename s_t, int VS = 16, typename elem_type_t>
static SFEM_INLINE int linear_elasticity_hessian_2d_esoa(
    const elem_type_t element_type,
        const ptrdiff_t nelements,
        const s_t *const *const RSTR coords,
        const s_t lmbda,
        const s_t mu,
        s_t *const *const RSTR matrix_streams
) {
  switch ((int)element_type) {
    case 3:
      return linear_elasticity_tri3_hessian_esoa<s_t, VS>(nelements, coords, lmbda, mu, matrix_streams);
    case 40:
      return linear_elasticity_quad4_hessian_esoa<s_t, VS>(nelements, coords, lmbda, mu, matrix_streams);
    case 400000:
      return linear_elasticity_proteus_quad4_hessian_esoa<s_t, VS>(nelements, coords, lmbda, mu, matrix_streams);
    default:
      return SFEM_FAILURE;
  }
}

template <typename s_t, int VS = 16, typename elem_type_t>
static SFEM_INLINE int linear_elasticity_hessian_3d_esoa(
    const elem_type_t element_type,
        const ptrdiff_t nelements,
        const s_t *const *const RSTR coords,
        const s_t lmbda,
        const s_t mu,
        s_t *const *const RSTR matrix_streams
) {
  switch ((int)element_type) {
    case 4:
      return linear_elasticity_tet4_hessian_esoa<s_t, VS>(nelements, coords, lmbda, mu, matrix_streams);
    case 10:
      return linear_elasticity_tet10_hessian_esoa<s_t, VS>(nelements, coords, lmbda, mu, matrix_streams);
    case 8:
      return linear_elasticity_hex8_hessian_esoa<s_t, VS>(nelements, coords, lmbda, mu, matrix_streams);
    case 100008:
      return linear_elasticity_proteus_hex8_hessian_esoa<s_t, VS>(nelements, coords, lmbda, mu, matrix_streams);
    default:
      return SFEM_FAILURE;
  }
}

template <typename s_t, int VS = 16, typename elem_type_t>
static SFEM_INLINE int linear_elasticity_hessian_2d_ecoords_soa(
    const elem_type_t element_type,
        const ptrdiff_t nelements,
        const s_t *const *const RSTR coords,
        const s_t lmbda,
        const s_t mu,
        s_t *const *const RSTR matrix_streams
) {
  switch ((int)element_type) {
    case 3:
      return linear_elasticity_tri3_hessian_ecoords_soa<s_t, VS>(nelements, coords, lmbda, mu, matrix_streams);
    case 40:
      return linear_elasticity_quad4_hessian_ecoords_soa<s_t, VS>(nelements, coords, lmbda, mu, matrix_streams);
    case 400000:
      return linear_elasticity_proteus_quad4_hessian_ecoords_soa<s_t, VS>(nelements, coords, lmbda, mu, matrix_streams);
    default:
      return SFEM_FAILURE;
  }
}

template <typename s_t, int VS = 16, typename elem_type_t>
static SFEM_INLINE int linear_elasticity_hessian_3d_ecoords_soa(
    const elem_type_t element_type,
        const ptrdiff_t nelements,
        const s_t *const *const RSTR coords,
        const s_t lmbda,
        const s_t mu,
        s_t *const *const RSTR matrix_streams
) {
  switch ((int)element_type) {
    case 4:
      return linear_elasticity_tet4_hessian_ecoords_soa<s_t, VS>(nelements, coords, lmbda, mu, matrix_streams);
    case 10:
      return linear_elasticity_tet10_hessian_ecoords_soa<s_t, VS>(nelements, coords, lmbda, mu, matrix_streams);
    case 8:
      return linear_elasticity_hex8_hessian_ecoords_soa<s_t, VS>(nelements, coords, lmbda, mu, matrix_streams);
    case 100008:
      return linear_elasticity_proteus_hex8_hessian_ecoords_soa<s_t, VS>(nelements, coords, lmbda, mu, matrix_streams);
    default:
      return SFEM_FAILURE;
  }
}

template <typename s_t, int VS = 16, typename elem_type_t>
static SFEM_INLINE int linear_elasticity_hessian_2d_egeometry_soa(
    const elem_type_t element_type,
        const ptrdiff_t nelements,
        const s_t *const *const RSTR adj,
        const s_t *const RSTR det,
        const s_t lmbda,
        const s_t mu,
        s_t *const *const RSTR matrix_streams
) {
  switch ((int)element_type) {
    case 3:
      return linear_elasticity_tri3_hessian_egeometry_soa<s_t, VS>(nelements, adj, det, lmbda, mu, matrix_streams);
    case 40:
      return linear_elasticity_quad4_hessian_egeometry_soa<s_t, VS>(nelements, adj, det, lmbda, mu, matrix_streams);
    case 400000:
      return linear_elasticity_proteus_quad4_hessian_egeometry_soa<s_t, VS>(nelements, adj, det, lmbda, mu, matrix_streams);
    default:
      return SFEM_FAILURE;
  }
}

template <typename s_t, int VS = 16, typename elem_type_t>
static SFEM_INLINE int linear_elasticity_hessian_3d_egeometry_soa(
    const elem_type_t element_type,
        const ptrdiff_t nelements,
        const s_t *const *const RSTR adj,
        const s_t *const RSTR det,
        const s_t lmbda,
        const s_t mu,
        s_t *const *const RSTR matrix_streams
) {
  switch ((int)element_type) {
    case 4:
      return linear_elasticity_tet4_hessian_egeometry_soa<s_t, VS>(nelements, adj, det, lmbda, mu, matrix_streams);
    case 10:
      return linear_elasticity_tet10_hessian_egeometry_soa<s_t, VS>(nelements, adj, det, lmbda, mu, matrix_streams);
    case 8:
      return linear_elasticity_hex8_hessian_egeometry_soa<s_t, VS>(nelements, adj, det, lmbda, mu, matrix_streams);
    case 100008:
      return linear_elasticity_proteus_hex8_hessian_egeometry_soa<s_t, VS>(nelements, adj, det, lmbda, mu, matrix_streams);
    default:
      return SFEM_FAILURE;
  }
}

} // namespace codegen
} // namespace sfem
