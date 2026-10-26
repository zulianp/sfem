#pragma once

#include <cstddef>

#include "mooney_rivlin_prony_proteus_quad4_element.hpp"
#include "mooney_rivlin_prony_quad4_element.hpp"
#include "mooney_rivlin_prony_tri3_element.hpp"
#include "mooney_rivlin_prony_hex8_element.hpp"
#include "mooney_rivlin_prony_proteus_hex8_element.hpp"
#include "mooney_rivlin_prony_tet10_element.hpp"
#include "mooney_rivlin_prony_tet4_element.hpp"

namespace sfem {
namespace codegen {

template <typename s_t, int VS, typename elem_type_t>
static SFEM_INLINE int mooney_rivlin_prony_energy_2d_esoa(
    const elem_type_t element_type,
        const ptrdiff_t nelements,
        const s_t *const *const RSTR coords,
        const s_t C01,
        const s_t C10,
        const s_t K,
        const s_t iso_relaxation_gamma,
        const s_t *const RSTR iso_history,
        const s_t *const *const RSTR u_streams,
        s_t *const RSTR values
) {
  switch ((int)element_type) {
    case 3:
      return mooney_rivlin_prony_tri3_energy_esoa<s_t, VS>(nelements, coords, C01, C10, K, iso_relaxation_gamma, iso_history, u_streams, values);
    case 40:
      return mooney_rivlin_prony_quad4_energy_esoa<s_t, VS>(nelements, coords, C01, C10, K, iso_relaxation_gamma, iso_history, u_streams, values);
    case 400000:
      return mooney_rivlin_prony_proteus_quad4_energy_esoa<s_t, VS>(nelements, coords, C01, C10, K, iso_relaxation_gamma, iso_history, u_streams, values);
    default:
      return SFEM_FAILURE;
  }
}

template <typename s_t, int VS, typename elem_type_t>
static SFEM_INLINE int mooney_rivlin_prony_energy_3d_esoa(
    const elem_type_t element_type,
        const ptrdiff_t nelements,
        const s_t *const *const RSTR coords,
        const s_t C01,
        const s_t C10,
        const s_t K,
        const s_t iso_relaxation_gamma,
        const s_t *const RSTR iso_history,
        const s_t *const *const RSTR u_streams,
        s_t *const RSTR values
) {
  switch ((int)element_type) {
    case 4:
      return mooney_rivlin_prony_tet4_energy_esoa<s_t, VS>(nelements, coords, C01, C10, K, iso_relaxation_gamma, iso_history, u_streams, values);
    case 10:
      return mooney_rivlin_prony_tet10_energy_esoa<s_t, VS>(nelements, coords, C01, C10, K, iso_relaxation_gamma, iso_history, u_streams, values);
    case 8:
      return mooney_rivlin_prony_hex8_energy_esoa<s_t, VS>(nelements, coords, C01, C10, K, iso_relaxation_gamma, iso_history, u_streams, values);
    case 100008:
      return mooney_rivlin_prony_proteus_hex8_energy_esoa<s_t, VS>(nelements, coords, C01, C10, K, iso_relaxation_gamma, iso_history, u_streams, values);
    default:
      return SFEM_FAILURE;
  }
}

template <typename s_t, int VS, typename elem_type_t>
static SFEM_INLINE int mooney_rivlin_prony_energy_2d_ecoords_soa(
    const elem_type_t element_type,
        const ptrdiff_t nelements,
        const s_t *const *const RSTR coords,
        const s_t C01,
        const s_t C10,
        const s_t K,
        const s_t iso_relaxation_gamma,
        const s_t *const RSTR iso_history,
        const s_t *const *const RSTR u_streams,
        s_t *const RSTR values
) {
  switch ((int)element_type) {
    case 3:
      return mooney_rivlin_prony_tri3_energy_ecoords_soa<s_t, VS>(nelements, coords, C01, C10, K, iso_relaxation_gamma, iso_history, u_streams, values);
    case 40:
      return mooney_rivlin_prony_quad4_energy_ecoords_soa<s_t, VS>(nelements, coords, C01, C10, K, iso_relaxation_gamma, iso_history, u_streams, values);
    case 400000:
      return mooney_rivlin_prony_proteus_quad4_energy_ecoords_soa<s_t, VS>(nelements, coords, C01, C10, K, iso_relaxation_gamma, iso_history, u_streams, values);
    default:
      return SFEM_FAILURE;
  }
}

template <typename s_t, int VS, typename elem_type_t>
static SFEM_INLINE int mooney_rivlin_prony_energy_3d_ecoords_soa(
    const elem_type_t element_type,
        const ptrdiff_t nelements,
        const s_t *const *const RSTR coords,
        const s_t C01,
        const s_t C10,
        const s_t K,
        const s_t iso_relaxation_gamma,
        const s_t *const RSTR iso_history,
        const s_t *const *const RSTR u_streams,
        s_t *const RSTR values
) {
  switch ((int)element_type) {
    case 4:
      return mooney_rivlin_prony_tet4_energy_ecoords_soa<s_t, VS>(nelements, coords, C01, C10, K, iso_relaxation_gamma, iso_history, u_streams, values);
    case 10:
      return mooney_rivlin_prony_tet10_energy_ecoords_soa<s_t, VS>(nelements, coords, C01, C10, K, iso_relaxation_gamma, iso_history, u_streams, values);
    case 8:
      return mooney_rivlin_prony_hex8_energy_ecoords_soa<s_t, VS>(nelements, coords, C01, C10, K, iso_relaxation_gamma, iso_history, u_streams, values);
    case 100008:
      return mooney_rivlin_prony_proteus_hex8_energy_ecoords_soa<s_t, VS>(nelements, coords, C01, C10, K, iso_relaxation_gamma, iso_history, u_streams, values);
    default:
      return SFEM_FAILURE;
  }
}

template <typename s_t, int VS, typename elem_type_t>
static SFEM_INLINE int mooney_rivlin_prony_energy_2d_egeometry_soa(
    const elem_type_t element_type,
        const ptrdiff_t nelements,
        const s_t *const *const RSTR adj,
        const s_t *const RSTR det,
        const s_t C01,
        const s_t C10,
        const s_t K,
        const s_t iso_relaxation_gamma,
        const s_t *const RSTR iso_history,
        const s_t *const *const RSTR u_streams,
        s_t *const RSTR values
) {
  switch ((int)element_type) {
    case 3:
      return mooney_rivlin_prony_tri3_energy_egeometry_soa<s_t, VS>(nelements, adj, det, C01, C10, K, iso_relaxation_gamma, iso_history, u_streams, values);
    case 40:
      return mooney_rivlin_prony_quad4_energy_egeometry_soa<s_t, VS>(nelements, adj, det, C01, C10, K, iso_relaxation_gamma, iso_history, u_streams, values);
    case 400000:
      return mooney_rivlin_prony_proteus_quad4_energy_egeometry_soa<s_t, VS>(nelements, adj, det, C01, C10, K, iso_relaxation_gamma, iso_history, u_streams, values);
    default:
      return SFEM_FAILURE;
  }
}

template <typename s_t, int VS, typename elem_type_t>
static SFEM_INLINE int mooney_rivlin_prony_energy_3d_egeometry_soa(
    const elem_type_t element_type,
        const ptrdiff_t nelements,
        const s_t *const *const RSTR adj,
        const s_t *const RSTR det,
        const s_t C01,
        const s_t C10,
        const s_t K,
        const s_t iso_relaxation_gamma,
        const s_t *const RSTR iso_history,
        const s_t *const *const RSTR u_streams,
        s_t *const RSTR values
) {
  switch ((int)element_type) {
    case 4:
      return mooney_rivlin_prony_tet4_energy_egeometry_soa<s_t, VS>(nelements, adj, det, C01, C10, K, iso_relaxation_gamma, iso_history, u_streams, values);
    case 10:
      return mooney_rivlin_prony_tet10_energy_egeometry_soa<s_t, VS>(nelements, adj, det, C01, C10, K, iso_relaxation_gamma, iso_history, u_streams, values);
    case 8:
      return mooney_rivlin_prony_hex8_energy_egeometry_soa<s_t, VS>(nelements, adj, det, C01, C10, K, iso_relaxation_gamma, iso_history, u_streams, values);
    case 100008:
      return mooney_rivlin_prony_proteus_hex8_energy_egeometry_soa<s_t, VS>(nelements, adj, det, C01, C10, K, iso_relaxation_gamma, iso_history, u_streams, values);
    default:
      return SFEM_FAILURE;
  }
}

template <typename s_t, int VS, typename elem_type_t>
static SFEM_INLINE int mooney_rivlin_prony_gradient_2d_esoa(
    const elem_type_t element_type,
        const ptrdiff_t nelements,
        const s_t *const *const RSTR coords,
        const s_t C01,
        const s_t C10,
        const s_t K,
        const s_t iso_relaxation_gamma,
        const s_t *const RSTR iso_history,
        const s_t *const *const RSTR u_streams,
        s_t *const *const RSTR out_streams
) {
  switch ((int)element_type) {
    case 3:
      return mooney_rivlin_prony_tri3_gradient_esoa<s_t, VS>(nelements, coords, C01, C10, K, iso_relaxation_gamma, iso_history, u_streams, out_streams);
    case 40:
      return mooney_rivlin_prony_quad4_gradient_esoa<s_t, VS>(nelements, coords, C01, C10, K, iso_relaxation_gamma, iso_history, u_streams, out_streams);
    case 400000:
      return mooney_rivlin_prony_proteus_quad4_gradient_esoa<s_t, VS>(nelements, coords, C01, C10, K, iso_relaxation_gamma, iso_history, u_streams, out_streams);
    default:
      return SFEM_FAILURE;
  }
}

template <typename s_t, int VS, typename elem_type_t>
static SFEM_INLINE int mooney_rivlin_prony_gradient_3d_esoa(
    const elem_type_t element_type,
        const ptrdiff_t nelements,
        const s_t *const *const RSTR coords,
        const s_t C01,
        const s_t C10,
        const s_t K,
        const s_t iso_relaxation_gamma,
        const s_t *const RSTR iso_history,
        const s_t *const *const RSTR u_streams,
        s_t *const *const RSTR out_streams
) {
  switch ((int)element_type) {
    case 4:
      return mooney_rivlin_prony_tet4_gradient_esoa<s_t, VS>(nelements, coords, C01, C10, K, iso_relaxation_gamma, iso_history, u_streams, out_streams);
    case 10:
      return mooney_rivlin_prony_tet10_gradient_esoa<s_t, VS>(nelements, coords, C01, C10, K, iso_relaxation_gamma, iso_history, u_streams, out_streams);
    case 8:
      return mooney_rivlin_prony_hex8_gradient_esoa<s_t, VS>(nelements, coords, C01, C10, K, iso_relaxation_gamma, iso_history, u_streams, out_streams);
    case 100008:
      return mooney_rivlin_prony_proteus_hex8_gradient_esoa<s_t, VS>(nelements, coords, C01, C10, K, iso_relaxation_gamma, iso_history, u_streams, out_streams);
    default:
      return SFEM_FAILURE;
  }
}

template <typename s_t, int VS, typename elem_type_t>
static SFEM_INLINE int mooney_rivlin_prony_gradient_2d_ecoords_soa(
    const elem_type_t element_type,
        const ptrdiff_t nelements,
        const s_t *const *const RSTR coords,
        const s_t C01,
        const s_t C10,
        const s_t K,
        const s_t iso_relaxation_gamma,
        const s_t *const RSTR iso_history,
        const s_t *const *const RSTR u_streams,
        s_t *const *const RSTR out_streams
) {
  switch ((int)element_type) {
    case 3:
      return mooney_rivlin_prony_tri3_gradient_ecoords_soa<s_t, VS>(nelements, coords, C01, C10, K, iso_relaxation_gamma, iso_history, u_streams, out_streams);
    case 40:
      return mooney_rivlin_prony_quad4_gradient_ecoords_soa<s_t, VS>(nelements, coords, C01, C10, K, iso_relaxation_gamma, iso_history, u_streams, out_streams);
    case 400000:
      return mooney_rivlin_prony_proteus_quad4_gradient_ecoords_soa<s_t, VS>(nelements, coords, C01, C10, K, iso_relaxation_gamma, iso_history, u_streams, out_streams);
    default:
      return SFEM_FAILURE;
  }
}

template <typename s_t, int VS, typename elem_type_t>
static SFEM_INLINE int mooney_rivlin_prony_gradient_3d_ecoords_soa(
    const elem_type_t element_type,
        const ptrdiff_t nelements,
        const s_t *const *const RSTR coords,
        const s_t C01,
        const s_t C10,
        const s_t K,
        const s_t iso_relaxation_gamma,
        const s_t *const RSTR iso_history,
        const s_t *const *const RSTR u_streams,
        s_t *const *const RSTR out_streams
) {
  switch ((int)element_type) {
    case 4:
      return mooney_rivlin_prony_tet4_gradient_ecoords_soa<s_t, VS>(nelements, coords, C01, C10, K, iso_relaxation_gamma, iso_history, u_streams, out_streams);
    case 10:
      return mooney_rivlin_prony_tet10_gradient_ecoords_soa<s_t, VS>(nelements, coords, C01, C10, K, iso_relaxation_gamma, iso_history, u_streams, out_streams);
    case 8:
      return mooney_rivlin_prony_hex8_gradient_ecoords_soa<s_t, VS>(nelements, coords, C01, C10, K, iso_relaxation_gamma, iso_history, u_streams, out_streams);
    case 100008:
      return mooney_rivlin_prony_proteus_hex8_gradient_ecoords_soa<s_t, VS>(nelements, coords, C01, C10, K, iso_relaxation_gamma, iso_history, u_streams, out_streams);
    default:
      return SFEM_FAILURE;
  }
}

template <typename s_t, int VS, typename elem_type_t>
static SFEM_INLINE int mooney_rivlin_prony_gradient_2d_egeometry_soa(
    const elem_type_t element_type,
        const ptrdiff_t nelements,
        const s_t *const *const RSTR adj,
        const s_t *const RSTR det,
        const s_t C01,
        const s_t C10,
        const s_t K,
        const s_t iso_relaxation_gamma,
        const s_t *const RSTR iso_history,
        const s_t *const *const RSTR u_streams,
        s_t *const *const RSTR out_streams
) {
  switch ((int)element_type) {
    case 3:
      return mooney_rivlin_prony_tri3_gradient_egeometry_soa<s_t, VS>(nelements, adj, det, C01, C10, K, iso_relaxation_gamma, iso_history, u_streams, out_streams);
    case 40:
      return mooney_rivlin_prony_quad4_gradient_egeometry_soa<s_t, VS>(nelements, adj, det, C01, C10, K, iso_relaxation_gamma, iso_history, u_streams, out_streams);
    case 400000:
      return mooney_rivlin_prony_proteus_quad4_gradient_egeometry_soa<s_t, VS>(nelements, adj, det, C01, C10, K, iso_relaxation_gamma, iso_history, u_streams, out_streams);
    default:
      return SFEM_FAILURE;
  }
}

template <typename s_t, int VS, typename elem_type_t>
static SFEM_INLINE int mooney_rivlin_prony_gradient_3d_egeometry_soa(
    const elem_type_t element_type,
        const ptrdiff_t nelements,
        const s_t *const *const RSTR adj,
        const s_t *const RSTR det,
        const s_t C01,
        const s_t C10,
        const s_t K,
        const s_t iso_relaxation_gamma,
        const s_t *const RSTR iso_history,
        const s_t *const *const RSTR u_streams,
        s_t *const *const RSTR out_streams
) {
  switch ((int)element_type) {
    case 4:
      return mooney_rivlin_prony_tet4_gradient_egeometry_soa<s_t, VS>(nelements, adj, det, C01, C10, K, iso_relaxation_gamma, iso_history, u_streams, out_streams);
    case 10:
      return mooney_rivlin_prony_tet10_gradient_egeometry_soa<s_t, VS>(nelements, adj, det, C01, C10, K, iso_relaxation_gamma, iso_history, u_streams, out_streams);
    case 8:
      return mooney_rivlin_prony_hex8_gradient_egeometry_soa<s_t, VS>(nelements, adj, det, C01, C10, K, iso_relaxation_gamma, iso_history, u_streams, out_streams);
    case 100008:
      return mooney_rivlin_prony_proteus_hex8_gradient_egeometry_soa<s_t, VS>(nelements, adj, det, C01, C10, K, iso_relaxation_gamma, iso_history, u_streams, out_streams);
    default:
      return SFEM_FAILURE;
  }
}

template <typename s_t, int VS, typename elem_type_t>
static SFEM_INLINE int mooney_rivlin_prony_hessian_2d_esoa(
    const elem_type_t element_type,
        const ptrdiff_t nelements,
        const s_t *const *const RSTR coords,
        const s_t C01,
        const s_t C10,
        const s_t K,
        const s_t iso_relaxation_gamma,
        const s_t *const RSTR iso_history,
        const s_t *const *const RSTR u_streams,
        s_t *const *const RSTR matrix_streams
) {
  switch ((int)element_type) {
    case 3:
      return mooney_rivlin_prony_tri3_hessian_esoa<s_t, VS>(nelements, coords, C01, C10, K, iso_relaxation_gamma, iso_history, u_streams, matrix_streams);
    case 40:
      return mooney_rivlin_prony_quad4_hessian_esoa<s_t, VS>(nelements, coords, C01, C10, K, iso_relaxation_gamma, iso_history, u_streams, matrix_streams);
    case 400000:
      return mooney_rivlin_prony_proteus_quad4_hessian_esoa<s_t, VS>(nelements, coords, C01, C10, K, iso_relaxation_gamma, iso_history, u_streams, matrix_streams);
    default:
      return SFEM_FAILURE;
  }
}

template <typename s_t, int VS, typename elem_type_t>
static SFEM_INLINE int mooney_rivlin_prony_hessian_3d_esoa(
    const elem_type_t element_type,
        const ptrdiff_t nelements,
        const s_t *const *const RSTR coords,
        const s_t C01,
        const s_t C10,
        const s_t K,
        const s_t iso_relaxation_gamma,
        const s_t *const RSTR iso_history,
        const s_t *const *const RSTR u_streams,
        s_t *const *const RSTR matrix_streams
) {
  switch ((int)element_type) {
    case 4:
      return mooney_rivlin_prony_tet4_hessian_esoa<s_t, VS>(nelements, coords, C01, C10, K, iso_relaxation_gamma, iso_history, u_streams, matrix_streams);
    case 10:
      return mooney_rivlin_prony_tet10_hessian_esoa<s_t, VS>(nelements, coords, C01, C10, K, iso_relaxation_gamma, iso_history, u_streams, matrix_streams);
    case 8:
      return mooney_rivlin_prony_hex8_hessian_esoa<s_t, VS>(nelements, coords, C01, C10, K, iso_relaxation_gamma, iso_history, u_streams, matrix_streams);
    case 100008:
      return mooney_rivlin_prony_proteus_hex8_hessian_esoa<s_t, VS>(nelements, coords, C01, C10, K, iso_relaxation_gamma, iso_history, u_streams, matrix_streams);
    default:
      return SFEM_FAILURE;
  }
}

template <typename s_t, int VS, typename elem_type_t>
static SFEM_INLINE int mooney_rivlin_prony_hessian_2d_ecoords_soa(
    const elem_type_t element_type,
        const ptrdiff_t nelements,
        const s_t *const *const RSTR coords,
        const s_t C01,
        const s_t C10,
        const s_t K,
        const s_t iso_relaxation_gamma,
        const s_t *const RSTR iso_history,
        const s_t *const *const RSTR u_streams,
        s_t *const *const RSTR matrix_streams
) {
  switch ((int)element_type) {
    case 3:
      return mooney_rivlin_prony_tri3_hessian_ecoords_soa<s_t, VS>(nelements, coords, C01, C10, K, iso_relaxation_gamma, iso_history, u_streams, matrix_streams);
    case 40:
      return mooney_rivlin_prony_quad4_hessian_ecoords_soa<s_t, VS>(nelements, coords, C01, C10, K, iso_relaxation_gamma, iso_history, u_streams, matrix_streams);
    case 400000:
      return mooney_rivlin_prony_proteus_quad4_hessian_ecoords_soa<s_t, VS>(nelements, coords, C01, C10, K, iso_relaxation_gamma, iso_history, u_streams, matrix_streams);
    default:
      return SFEM_FAILURE;
  }
}

template <typename s_t, int VS, typename elem_type_t>
static SFEM_INLINE int mooney_rivlin_prony_hessian_3d_ecoords_soa(
    const elem_type_t element_type,
        const ptrdiff_t nelements,
        const s_t *const *const RSTR coords,
        const s_t C01,
        const s_t C10,
        const s_t K,
        const s_t iso_relaxation_gamma,
        const s_t *const RSTR iso_history,
        const s_t *const *const RSTR u_streams,
        s_t *const *const RSTR matrix_streams
) {
  switch ((int)element_type) {
    case 4:
      return mooney_rivlin_prony_tet4_hessian_ecoords_soa<s_t, VS>(nelements, coords, C01, C10, K, iso_relaxation_gamma, iso_history, u_streams, matrix_streams);
    case 10:
      return mooney_rivlin_prony_tet10_hessian_ecoords_soa<s_t, VS>(nelements, coords, C01, C10, K, iso_relaxation_gamma, iso_history, u_streams, matrix_streams);
    case 8:
      return mooney_rivlin_prony_hex8_hessian_ecoords_soa<s_t, VS>(nelements, coords, C01, C10, K, iso_relaxation_gamma, iso_history, u_streams, matrix_streams);
    case 100008:
      return mooney_rivlin_prony_proteus_hex8_hessian_ecoords_soa<s_t, VS>(nelements, coords, C01, C10, K, iso_relaxation_gamma, iso_history, u_streams, matrix_streams);
    default:
      return SFEM_FAILURE;
  }
}

template <typename s_t, int VS, typename elem_type_t>
static SFEM_INLINE int mooney_rivlin_prony_hessian_2d_egeometry_soa(
    const elem_type_t element_type,
        const ptrdiff_t nelements,
        const s_t *const *const RSTR adj,
        const s_t *const RSTR det,
        const s_t C01,
        const s_t C10,
        const s_t K,
        const s_t iso_relaxation_gamma,
        const s_t *const RSTR iso_history,
        const s_t *const *const RSTR u_streams,
        s_t *const *const RSTR matrix_streams
) {
  switch ((int)element_type) {
    case 3:
      return mooney_rivlin_prony_tri3_hessian_egeometry_soa<s_t, VS>(nelements, adj, det, C01, C10, K, iso_relaxation_gamma, iso_history, u_streams, matrix_streams);
    case 40:
      return mooney_rivlin_prony_quad4_hessian_egeometry_soa<s_t, VS>(nelements, adj, det, C01, C10, K, iso_relaxation_gamma, iso_history, u_streams, matrix_streams);
    case 400000:
      return mooney_rivlin_prony_proteus_quad4_hessian_egeometry_soa<s_t, VS>(nelements, adj, det, C01, C10, K, iso_relaxation_gamma, iso_history, u_streams, matrix_streams);
    default:
      return SFEM_FAILURE;
  }
}

template <typename s_t, int VS, typename elem_type_t>
static SFEM_INLINE int mooney_rivlin_prony_hessian_3d_egeometry_soa(
    const elem_type_t element_type,
        const ptrdiff_t nelements,
        const s_t *const *const RSTR adj,
        const s_t *const RSTR det,
        const s_t C01,
        const s_t C10,
        const s_t K,
        const s_t iso_relaxation_gamma,
        const s_t *const RSTR iso_history,
        const s_t *const *const RSTR u_streams,
        s_t *const *const RSTR matrix_streams
) {
  switch ((int)element_type) {
    case 4:
      return mooney_rivlin_prony_tet4_hessian_egeometry_soa<s_t, VS>(nelements, adj, det, C01, C10, K, iso_relaxation_gamma, iso_history, u_streams, matrix_streams);
    case 10:
      return mooney_rivlin_prony_tet10_hessian_egeometry_soa<s_t, VS>(nelements, adj, det, C01, C10, K, iso_relaxation_gamma, iso_history, u_streams, matrix_streams);
    case 8:
      return mooney_rivlin_prony_hex8_hessian_egeometry_soa<s_t, VS>(nelements, adj, det, C01, C10, K, iso_relaxation_gamma, iso_history, u_streams, matrix_streams);
    case 100008:
      return mooney_rivlin_prony_proteus_hex8_hessian_egeometry_soa<s_t, VS>(nelements, adj, det, C01, C10, K, iso_relaxation_gamma, iso_history, u_streams, matrix_streams);
    default:
      return SFEM_FAILURE;
  }
}

} // namespace codegen
} // namespace sfem
