// Dispatch for vector Laplacian matrix-free apply. Precomputed FFF must match
// smesh::FFF::create_AoS / fill_fff in external/smesh/src/frontend/smesh_kernel_data.cpp
// (hex8_fff_fill, sshex8_macro_fff_fill for semi-structured types).

#include "vector_laplacian.hpp"

#include "hex8_vector_laplacian.hpp"
#include "sfem_defs.hpp"
#include "sfem_team.h"
#include "smesh_semistructured.hpp"
#include "sshex8_vector_laplacian.hpp"

template <bool TEAM>
static int vector_laplacian_apply_dispatch(smesh::ElemType              element_type,
                           const ptrdiff_t              nelements,
                           const ptrdiff_t              nnodes,
                           idx_t **const SFEM_RESTRICT  elements,
                           geom_t **const SFEM_RESTRICT points,
                           const int                    vector_size,
                           const ptrdiff_t              stride,
                           real_t **const SFEM_RESTRICT u,
                           real_t **const SFEM_RESTRICT values) {
    if (sfem::is_semistructured_type(element_type)) {
        SFEM_ERROR(
                "vector_laplacian_apply: semi-structured vector Laplacian requires precomputed FFF; use "
                "vector_laplacian_apply_opt with smesh::FFF::create_AoS (see smesh_kernel_data.cpp).\n");
        return SFEM_FAILURE;
    }

    switch (element_type) {
        case smesh::HEX8: {
            return SFEM_TEAM_KERNEL(TEAM, affine_hex8_vector_laplacian_apply)(
                    nelements, nnodes, elements, points, vector_size, stride, u, values);
        }
        default: {
            SFEM_ERROR("vector_laplacian_apply not implemented for type %s\n", sfem::type_to_string(element_type));
        }
    }

    return SFEM_FAILURE;
}

int vector_laplacian_apply(smesh::ElemType              element_type,
                           const ptrdiff_t              nelements,
                           const ptrdiff_t              nnodes,
                           idx_t **const SFEM_RESTRICT  elements,
                           geom_t **const SFEM_RESTRICT points,
                           const int                    vector_size,
                           const ptrdiff_t              stride,
                           real_t **const SFEM_RESTRICT u,
                           real_t **const SFEM_RESTRICT values) {
    return vector_laplacian_apply_dispatch<false>(element_type, nelements, nnodes, elements, points, vector_size, stride, u, values);
}

int vector_laplacian_apply_team(smesh::ElemType              element_type,
                           const ptrdiff_t              nelements,
                           const ptrdiff_t              nnodes,
                           idx_t **const SFEM_RESTRICT  elements,
                           geom_t **const SFEM_RESTRICT points,
                           const int                    vector_size,
                           const ptrdiff_t              stride,
                           real_t **const SFEM_RESTRICT u,
                           real_t **const SFEM_RESTRICT values) {
    return vector_laplacian_apply_dispatch<true>(element_type, nelements, nnodes, elements, points, vector_size, stride, u, values);
}

template <bool TEAM>
static int vector_laplacian_apply_opt_dispatch(smesh::ElemType                     element_type,
                               const ptrdiff_t                     nelements,
                               idx_t **const SFEM_RESTRICT           elements,
                               const jacobian_t *const SFEM_RESTRICT fff,
                               const int                           vector_size,
                               const ptrdiff_t                     stride,
                               real_t **const SFEM_RESTRICT          u,
                               real_t **const SFEM_RESTRICT          values) {
    if (sfem::is_semistructured_type(element_type)) {
        const int level = smesh::semistructured_level(element_type);
        // No team form: as a function called from its region, the semi-structured kernel measured
        // 1-3% slower than with the region inside it (Grace, 12.5M dofs, levels 4 and 8).
        if constexpr (TEAM) {
            SFEM_ERROR("vector_laplacian_apply_opt_team: no semi-structured kernel\n");
            return SFEM_FAILURE;
        }
        return affine_sshex8_vector_laplacian_apply_fff(
                level, nelements, elements, fff, vector_size, stride, u, values);
    }

    switch (element_type) {
        case smesh::HEX8: {
            return SFEM_TEAM_KERNEL(TEAM, affine_hex8_vector_laplacian_apply_fff)(
                    nelements, elements, fff, vector_size, stride, u, values);
        }
        default: {
            SFEM_ERROR("vector_laplacian_apply_opt not implemented for type %s\n", sfem::type_to_string(element_type));
        }
    }

    return SFEM_FAILURE;
}

int vector_laplacian_apply_opt(smesh::ElemType                     element_type,
                               const ptrdiff_t                     nelements,
                               idx_t **const SFEM_RESTRICT           elements,
                               const jacobian_t *const SFEM_RESTRICT fff,
                               const int                           vector_size,
                               const ptrdiff_t                     stride,
                               real_t **const SFEM_RESTRICT          u,
                               real_t **const SFEM_RESTRICT          values) {
    return vector_laplacian_apply_opt_dispatch<false>(element_type, nelements, elements, fff, vector_size, stride, u, values);
}

int vector_laplacian_apply_opt_team(smesh::ElemType                     element_type,
                               const ptrdiff_t                     nelements,
                               idx_t **const SFEM_RESTRICT           elements,
                               const jacobian_t *const SFEM_RESTRICT fff,
                               const int                           vector_size,
                               const ptrdiff_t                     stride,
                               real_t **const SFEM_RESTRICT          u,
                               real_t **const SFEM_RESTRICT          values) {
    return vector_laplacian_apply_opt_dispatch<true>(element_type, nelements, elements, fff, vector_size, stride, u, values);
}
