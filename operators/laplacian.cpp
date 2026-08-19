#include "laplacian.hpp"

#include "tet10_laplacian.hpp"
#include "tet4_laplacian.hpp"
#include "tri3_laplacian.hpp"
#include "tri6_laplacian.hpp"

#include "macro_tet4_laplacian.hpp"
#include "macro_tri3_laplacian.hpp"

#include "hex8_laplacian.hpp"
#include "spectral_hex_laplacian.hpp"
#include "sshex8_laplacian.hpp"

#include "sfem_defs.hpp"
#include "smesh_ssquad4.hpp"

#include <mpi.h>
#include <stdio.h>

namespace {

    static SFEM_INLINE void quad4_laplacian_apply_micro(const idx_t *const SFEM_RESTRICT  ev,
                                                        geom_t **const SFEM_RESTRICT      points,
                                                        const real_t *const SFEM_RESTRICT u,
                                                        real_t *const SFEM_RESTRICT       element_vector) {
        static constexpr real_t q     = real_t(0.57735026918962576450914878050195746);
        static constexpr real_t qp[2] = {-q, q};

        const real_t x[4]  = {real_t(points[0][ev[0]]), real_t(points[0][ev[1]]), real_t(points[0][ev[2]]), real_t(points[0][ev[3]])};
        const real_t y[4]  = {real_t(points[1][ev[0]]), real_t(points[1][ev[1]]), real_t(points[1][ev[2]]), real_t(points[1][ev[3]])};
        const real_t ue[4] = {u[ev[0]], u[ev[1]], u[ev[2]], u[ev[3]]};

        element_vector[0] = 0;
        element_vector[1] = 0;
        element_vector[2] = 0;
        element_vector[3] = 0;

        for (int iy = 0; iy < 2; ++iy) {
            const real_t eta = qp[iy];
            for (int ix = 0; ix < 2; ++ix) {
                const real_t xi = qp[ix];

                const real_t dndxi[4] = {-(real_t(1) - eta) * real_t(0.25),
                                          (real_t(1) - eta) * real_t(0.25),
                                          (real_t(1) + eta) * real_t(0.25),
                                          -(real_t(1) + eta) * real_t(0.25)};
                const real_t dndeta[4] = {-(real_t(1) - xi) * real_t(0.25),
                                           -(real_t(1) + xi) * real_t(0.25),
                                           (real_t(1) + xi) * real_t(0.25),
                                           (real_t(1) - xi) * real_t(0.25)};

                real_t dx_dxi  = 0;
                real_t dx_deta = 0;
                real_t dy_dxi  = 0;
                real_t dy_deta = 0;
                for (int a = 0; a < 4; ++a) {
                    dx_dxi += x[a] * dndxi[a];
                    dx_deta += x[a] * dndeta[a];
                    dy_dxi += y[a] * dndxi[a];
                    dy_deta += y[a] * dndeta[a];
                }

                const real_t det     = dx_dxi * dy_deta - dx_deta * dy_dxi;
                const real_t inv_det = real_t(1) / det;

                real_t gx[4];
                real_t gy[4];
                real_t gux = 0;
                real_t guy = 0;
                for (int a = 0; a < 4; ++a) {
                    gx[a] = (dy_deta * dndxi[a] - dy_dxi * dndeta[a]) * inv_det;
                    gy[a] = (-dx_deta * dndxi[a] + dx_dxi * dndeta[a]) * inv_det;
                    gux += gx[a] * ue[a];
                    guy += gy[a] * ue[a];
                }

                for (int a = 0; a < 4; ++a) {
                    element_vector[a] += det * (gx[a] * gux + gy[a] * guy);
                }
            }
        }
    }

    static SFEM_INLINE void quad4_laplacian_diag_micro(const idx_t *const SFEM_RESTRICT ev,
                                                       geom_t **const SFEM_RESTRICT     points,
                                                       real_t *const SFEM_RESTRICT      element_diag) {
        static constexpr real_t q     = real_t(0.57735026918962576450914878050195746);
        static constexpr real_t qp[2] = {-q, q};

        const real_t x[4] = {real_t(points[0][ev[0]]), real_t(points[0][ev[1]]), real_t(points[0][ev[2]]), real_t(points[0][ev[3]])};
        const real_t y[4] = {real_t(points[1][ev[0]]), real_t(points[1][ev[1]]), real_t(points[1][ev[2]]), real_t(points[1][ev[3]])};

        element_diag[0] = 0;
        element_diag[1] = 0;
        element_diag[2] = 0;
        element_diag[3] = 0;

        for (int iy = 0; iy < 2; ++iy) {
            const real_t eta = qp[iy];
            for (int ix = 0; ix < 2; ++ix) {
                const real_t xi = qp[ix];

                const real_t dndxi[4] = {-(real_t(1) - eta) * real_t(0.25),
                                          (real_t(1) - eta) * real_t(0.25),
                                          (real_t(1) + eta) * real_t(0.25),
                                          -(real_t(1) + eta) * real_t(0.25)};
                const real_t dndeta[4] = {-(real_t(1) - xi) * real_t(0.25),
                                           -(real_t(1) + xi) * real_t(0.25),
                                           (real_t(1) + xi) * real_t(0.25),
                                           (real_t(1) - xi) * real_t(0.25)};

                real_t dx_dxi  = 0;
                real_t dx_deta = 0;
                real_t dy_dxi  = 0;
                real_t dy_deta = 0;
                for (int a = 0; a < 4; ++a) {
                    dx_dxi += x[a] * dndxi[a];
                    dx_deta += x[a] * dndeta[a];
                    dy_dxi += y[a] * dndxi[a];
                    dy_deta += y[a] * dndeta[a];
                }

                const real_t det     = dx_dxi * dy_deta - dx_deta * dy_dxi;
                const real_t inv_det = real_t(1) / det;

                for (int a = 0; a < 4; ++a) {
                    const real_t gx = (dy_deta * dndxi[a] - dy_dxi * dndeta[a]) * inv_det;
                    const real_t gy = (-dx_deta * dndxi[a] + dx_dxi * dndeta[a]) * inv_det;
                    element_diag[a] += det * (gx * gx + gy * gy);
                }
            }
        }
    }

    int quad4_laplacian_apply_isoparametric(const ptrdiff_t                   nelements,
                                            idx_t **const SFEM_RESTRICT       elements,
                                            geom_t **const SFEM_RESTRICT      points,
                                            const real_t *const SFEM_RESTRICT u,
                                            real_t *const SFEM_RESTRICT       values) {
#pragma omp parallel for schedule(static)
        for (ptrdiff_t e = 0; e < nelements; ++e) {
            const idx_t ev[4] = {elements[0][e], elements[1][e], elements[2][e], elements[3][e]};
            real_t      element_vector[4];
            quad4_laplacian_apply_micro(ev, points, u, element_vector);
            for (int a = 0; a < 4; ++a) {
#pragma omp atomic update
                values[ev[a]] += element_vector[a];
            }
        }

        return SFEM_SUCCESS;
    }

    int quad4_laplacian_diag_isoparametric(const ptrdiff_t              nelements,
                                           idx_t **const SFEM_RESTRICT  elements,
                                           geom_t **const SFEM_RESTRICT points,
                                           real_t *const SFEM_RESTRICT  values) {
#pragma omp parallel for schedule(static)
        for (ptrdiff_t e = 0; e < nelements; ++e) {
            const idx_t ev[4] = {elements[0][e], elements[1][e], elements[2][e], elements[3][e]};
            real_t      element_diag[4];
            quad4_laplacian_diag_micro(ev, points, element_diag);
            for (int a = 0; a < 4; ++a) {
#pragma omp atomic update
                values[ev[a]] += element_diag[a];
            }
        }

        return SFEM_SUCCESS;
    }

    int ssquad4_laplacian_apply_isoparametric(const int                         level,
                                              const ptrdiff_t                   nelements,
                                              idx_t **const SFEM_RESTRICT       elements,
                                              geom_t **const SFEM_RESTRICT      points,
                                              const real_t *const SFEM_RESTRICT u,
                                              real_t *const SFEM_RESTRICT       values) {
#pragma omp parallel for schedule(static)
        for (ptrdiff_t e = 0; e < nelements; ++e) {
            for (int yi = 0; yi < level; ++yi) {
                for (int xi = 0; xi < level; ++xi) {
                    const idx_t ev[4] = {elements[smesh::ssquad4_lidx(level, xi, yi)][e],
                                         elements[smesh::ssquad4_lidx(level, xi + 1, yi)][e],
                                         elements[smesh::ssquad4_lidx(level, xi + 1, yi + 1)][e],
                                         elements[smesh::ssquad4_lidx(level, xi, yi + 1)][e]};
                    real_t      element_vector[4];
                    quad4_laplacian_apply_micro(ev, points, u, element_vector);
                    for (int a = 0; a < 4; ++a) {
#pragma omp atomic update
                        values[ev[a]] += element_vector[a];
                    }
                }
            }
        }

        return SFEM_SUCCESS;
    }

    int ssquad4_laplacian_diag_isoparametric(const int                    level,
                                             const ptrdiff_t              nelements,
                                             idx_t **const SFEM_RESTRICT  elements,
                                             geom_t **const SFEM_RESTRICT points,
                                             real_t *const SFEM_RESTRICT  values) {
#pragma omp parallel for schedule(static)
        for (ptrdiff_t e = 0; e < nelements; ++e) {
            for (int yi = 0; yi < level; ++yi) {
                for (int xi = 0; xi < level; ++xi) {
                    const idx_t ev[4] = {elements[smesh::ssquad4_lidx(level, xi, yi)][e],
                                         elements[smesh::ssquad4_lidx(level, xi + 1, yi)][e],
                                         elements[smesh::ssquad4_lidx(level, xi + 1, yi + 1)][e],
                                         elements[smesh::ssquad4_lidx(level, xi, yi + 1)][e]};
                    real_t      element_diag[4];
                    quad4_laplacian_diag_micro(ev, points, element_diag);
                    for (int a = 0; a < 4; ++a) {
#pragma omp atomic update
                        values[ev[a]] += element_diag[a];
                    }
                }
            }
        }

        return SFEM_SUCCESS;
    }

}  // namespace

int laplacian_is_opt(smesh::ElemType element_type) {
    return element_type == smesh::TRI3 || element_type == smesh::TET10 || element_type == smesh::TET4 ||
           element_type == smesh::MACRO_TET4 || element_type == smesh::MACRO_TRI3 ||
           (sfem::is_semistructured_type(element_type) && smesh::is_hex_ss_family(element_type));
}

int laplacian_assemble_value(smesh::ElemType                   element_type,
                             const ptrdiff_t                   nelements,
                             const ptrdiff_t                   nnodes,
                             idx_t **const SFEM_RESTRICT       elements,
                             geom_t **const SFEM_RESTRICT      points,
                             const real_t *const SFEM_RESTRICT u,
                             real_t *const SFEM_RESTRICT       value) {
    if (sfem::is_semistructured_type(element_type)) {
        SFEM_ERROR("laplacian_assemble_value not implemented for semi-structured element type %s\n",
                   sfem::type_to_string(element_type));
        return SFEM_FAILURE;
    }

    switch (element_type) {
        case smesh::TRI3: {
            return tri3_laplacian_assemble_value(nelements, nnodes, elements, points, u, value);
        }
        case smesh::TRI6: {
            return tri6_laplacian_assemble_value(nelements, nnodes, elements, points, u, value);
        }
        case smesh::TET4: {
            return tet4_laplacian_assemble_value(nelements, nnodes, elements, points, u, value);
        }
        case smesh::TET10: {
            return tet10_laplacian_assemble_value(nelements, nnodes, elements, points, u, value);
        }
        default: {
            SFEM_ERROR("laplacian_assemble_value not implemented for type %s\n", sfem::type_to_string(element_type));
        }
    }

    return SFEM_FAILURE;
}

int laplacian_apply(smesh::ElemType                   element_type,
                    const ptrdiff_t                   nelements,
                    const ptrdiff_t                   nnodes,
                    idx_t **const SFEM_RESTRICT       elements,
                    geom_t **const SFEM_RESTRICT      points,
                    const real_t *const SFEM_RESTRICT u,
                    real_t *const SFEM_RESTRICT       values) {
    if (sfem::is_semistructured_type(element_type)) {
        const int level = smesh::semistructured_level(element_type);
        if (smesh::is_hex_ss_family(element_type)) {
            return sshex8_laplacian_apply(level, nelements, elements, points, u, values);
        }
        if (smesh::is_quad_ss_family(element_type)) {
            return ssquad4_laplacian_apply_isoparametric(level, nelements, elements, points, u, values);
        }

        SFEM_ERROR("laplacian_apply not implemented for semi-structured element type %s\n",
                   sfem::type_to_string(element_type));
        return SFEM_FAILURE;
    }

    switch (element_type) {
        case smesh::TRI3: {
            return tri3_laplacian_apply(nelements, nnodes, elements, points, u, values);
        }
        case smesh::TRI6: {
            return tri6_laplacian_apply(nelements, nnodes, elements, points, u, values);
        }
        case smesh::TET4: {
            return tet4_laplacian_apply(nelements, nnodes, elements, points, u, values);
        }
        case smesh::TET10: {
            return tet10_laplacian_apply(nelements, nnodes, elements, points, u, values);
        }
        case smesh::MACRO_TET4: {
            return macro_tet4_laplacian_apply(nelements, nnodes, elements, points, u, values);
        }
        case smesh::MACRO_TRI3: {
            return macro_tri3_laplacian_apply(nelements, nnodes, elements, points, u, values);
        }
        case smesh::HEX8: {
            return hex8_laplacian_apply(nelements, nnodes, elements, points, u, values);
        }
        case smesh::QUAD4: {
            return quad4_laplacian_apply_isoparametric(nelements, elements, points, u, values);
        }
        default: {
            SFEM_ERROR("laplacian_apply not implemented for type %s\n", sfem::type_to_string(element_type));
        }
    }

    return SFEM_FAILURE;
}

int laplacian_assemble_gradient(smesh::ElemType                   element_type,
                                const ptrdiff_t                   nelements,
                                const ptrdiff_t                   nnodes,
                                idx_t **const SFEM_RESTRICT       elements,
                                geom_t **const SFEM_RESTRICT      points,
                                const real_t *const SFEM_RESTRICT u,
                                real_t *const SFEM_RESTRICT       values) {
    return laplacian_apply(element_type, nelements, nnodes, elements, points, u, values);
}

int laplacian_crs(smesh::ElemType                    element_type,
                  const ptrdiff_t                    nelements,
                  const ptrdiff_t                    nnodes,
                  idx_t **const SFEM_RESTRICT        elements,
                  geom_t **const SFEM_RESTRICT       points,
                  const count_t *const SFEM_RESTRICT rowptr,
                  const idx_t *const SFEM_RESTRICT   colidx,
                  real_t *const SFEM_RESTRICT        values) {
    if (sfem::is_semistructured_type(element_type)) {
        SFEM_ERROR("laplacian_crs not implemented for semi-structured element type %s\n",
                   sfem::type_to_string(element_type));
        return SFEM_FAILURE;
    }

    switch (element_type) {
        case smesh::TRI3: {
            return tri3_laplacian_crs(nelements, nnodes, elements, points, rowptr, colidx, values);
        }
        case smesh::TRI6: {
            return tri6_laplacian_crs(nelements, nnodes, elements, points, rowptr, colidx, values);
        }
        case smesh::TET4: {
            return tet4_laplacian_crs(nelements, nnodes, elements, points, rowptr, colidx, values);
        }
        case smesh::HEX8: {
            return hex8_laplacian_crs(nelements, nnodes, elements, points, rowptr, colidx, values);
        }
        case smesh::TET10: {
            return tet10_laplacian_crs(nelements, nnodes, elements, points, rowptr, colidx, values);
        }
        case smesh::MACRO_TET4: {
            return macro_tet4_laplacian_crs(nelements, nnodes, elements, points, rowptr, colidx, values);
        }
        case smesh::MACRO_TRI3: {
            return macro_tri3_laplacian_crs(nelements, nnodes, elements, points, rowptr, colidx, values);
        }
        default: {
            SFEM_ERROR("laplacian_crs not implemented for type %s\n", sfem::type_to_string(element_type));
        }
    }

    return SFEM_FAILURE;
}

int laplacian_diag(smesh::ElemType              element_type,
                   const ptrdiff_t              nelements,
                   const ptrdiff_t              nnodes,
                   idx_t **const SFEM_RESTRICT  elements,
                   geom_t **const SFEM_RESTRICT points,
                   real_t *const SFEM_RESTRICT  values) {
    if (sfem::is_semistructured_type(element_type)) {
        const int level = smesh::semistructured_level(element_type);
        if (smesh::is_hex_ss_family(element_type)) {
            return affine_sshex8_laplacian_diag(level, nelements, elements, points, values);
        }
        if (smesh::is_quad_ss_family(element_type)) {
            return ssquad4_laplacian_diag_isoparametric(level, nelements, elements, points, values);
        }

        SFEM_ERROR("laplacian_diag not implemented for semi-structured element type %s\n",
                   sfem::type_to_string(element_type));
        return SFEM_FAILURE;
    }

    switch (element_type) {
        case smesh::TRI3: {
            return tri3_laplacian_diag(nelements, nnodes, elements, points, values);
        }
        case smesh::TRI6: {
            return tri6_laplacian_diag(nelements, nnodes, elements, points, values);
        }
        case smesh::TET4: {
            return tet4_laplacian_diag(nelements, nnodes, elements, points, values);
        }
        case smesh::HEX8: {
            return hex8_laplacian_diag(nelements, nnodes, elements, points, values);
        }
        case smesh::QUAD4: {
            return quad4_laplacian_diag_isoparametric(nelements, elements, points, values);
        }
        case smesh::TET10: {
            return tet10_laplacian_diag(nelements, nnodes, elements, points, values);
        }
        case smesh::MACRO_TET4: {
            return macro_tet4_laplacian_diag(nelements, nnodes, elements, points, values);
        }
        case smesh::MACRO_TRI3: {
            return macro_tri3_laplacian_diag(nelements, nnodes, elements, points, values);
        }
        default: {
            SFEM_ERROR("laplacian_diag not implemented for type %s\n", sfem::type_to_string(element_type));
        }
    }

    return SFEM_FAILURE;
}

int laplacian_apply_opt(smesh::ElemType                       element_type,
                        const ptrdiff_t                       nelements,
                        idx_t **const SFEM_RESTRICT           elements,
                        const jacobian_t *const SFEM_RESTRICT fff,
                        const real_t *const SFEM_RESTRICT     u,
                        real_t *const SFEM_RESTRICT           values) {
    if (sfem::is_semistructured_type(element_type)) {
        const int level = smesh::semistructured_level(element_type);
        if (smesh::is_hex_ss_family(element_type)) {
            return affine_sshex8_laplacian_stencil_apply_fff(level, nelements, elements, fff, u, values);
        }

        SFEM_ERROR("laplacian_apply_opt not implemented for semi-structured element type %s\n",
                   sfem::type_to_string(element_type));
        return SFEM_FAILURE;
    }

    switch (element_type) {
        case smesh::TRI3: {
            return tri3_laplacian_apply_opt(nelements, elements, fff, u, values);
        }
        case smesh::TRI6: {
            return tri6_laplacian_apply_opt(nelements, elements, fff, u, values);
        }
        case smesh::TET4: {
            return tet4_laplacian_apply_opt(nelements, elements, fff, u, values);
        }
        case smesh::HEX8: {
            return hex8_laplacian_apply_opt(nelements, elements, fff, u, values);
        }
        case smesh::TET10: {
            return tet10_laplacian_apply_opt(nelements, elements, fff, u, values);
        }
        case smesh::MACRO_TET4: {
            return macro_tet4_laplacian_apply_opt(nelements, elements, fff, u, values);
        }
        case smesh::MACRO_TRI3: {
            return macro_tri3_laplacian_apply_opt(nelements, elements, fff, u, values);
        }
        default: {
            SFEM_ERROR("laplacian_apply_opt not implemented for type %s\n", sfem::type_to_string(element_type));
        }
    }

    return SFEM_FAILURE;
}

int laplacian_crs_sym(smesh::ElemType                    element_type,
                      const ptrdiff_t                    nelements,
                      const ptrdiff_t                    nnodes,
                      idx_t **const SFEM_RESTRICT        elements,
                      geom_t **const SFEM_RESTRICT       points,
                      const count_t *const SFEM_RESTRICT rowptr,
                      const idx_t *const SFEM_RESTRICT   colidx,
                      real_t *const SFEM_RESTRICT        diag,
                      real_t *const SFEM_RESTRICT        offdiag) {
    if (sfem::is_semistructured_type(element_type)) {
        SFEM_ERROR("laplacian_crs_sym not implemented for semi-structured element type %s\n",
                   sfem::type_to_string(element_type));
        return SFEM_FAILURE;
    }

    switch (element_type) {
        case smesh::HEX8: {
            return hex8_laplacian_crs_sym(nelements, nnodes, elements, points, rowptr, colidx, diag, offdiag);
        }
        default: {
            SFEM_ERROR("laplacian_crs_sym not implemented for type %s\n", sfem::type_to_string(element_type));
        }
    }

    return SFEM_FAILURE;
}
