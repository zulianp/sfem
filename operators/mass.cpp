#include "mass.hpp"

#include <assert.h>
#include <math.h>
#include <stdio.h>

#include <mpi.h>


#include "sortreduce.hpp"

#include "sfem_vec.hpp"

#include "sfem_defs.hpp"

#include "beam2_mass.hpp"
#include "hex8_mass.hpp"
#include "quadshell4_mass.hpp"
#include "tet10_mass.hpp"
#include "tet4_mass.hpp"
#include "tri3_mass.hpp"
#include "tri6_mass.hpp"
#include "trishell3_mass.hpp"
#include "sshex8_mass.hpp"

static void hex27_assemble_lobatto_lumped_mass(const ptrdiff_t              nelements,
                                               const ptrdiff_t              nnodes,
                                               idx_t **const SFEM_RESTRICT  elements,
                                               geom_t **const SFEM_RESTRICT points,
                                               const bool                   cartesian_ordering,
                                               real_t *const SFEM_RESTRICT  values) {
    (void)nnodes;

    static const int sfem_hex27_to_cartesian[27] = {
            0, 2, 8, 6, 18, 20, 26, 24, 1, 5, 7, 3, 19, 23,
            25, 21, 9, 11, 17, 15, 10, 14, 16, 12, 4, 22, 13,
    };

    static const int cartesian_to_sfem_hex27[27] = {
            0, 8, 1, 11, 24, 9, 3, 10, 2, 16, 20, 17, 23, 26,
            21, 19, 22, 18, 4, 12, 5, 15, 25, 13, 7, 14, 6,
    };

    static const int identity_hex27[27] = {
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13,
            14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26,
    };

    const int *const SFEM_RESTRICT local_to_cartesian = cartesian_ordering ? identity_hex27 : sfem_hex27_to_cartesian;
    const int *const SFEM_RESTRICT cartesian_to_local = cartesian_ordering ? identity_hex27 : cartesian_to_sfem_hex27;

    static const real_t shape[3][3] = {
            {1, 0, 0},
            {0, 1, 0},
            {0, 0, 1},
    };

    static const real_t grad[3][3] = {
            {-1.5, 2, -0.5},
            {-0.5, 0, 0.5},
            {0.5, -2, 1.5},
    };

    static const real_t weight[3] = {1.0 / 3.0, 4.0 / 3.0, 1.0 / 3.0};

    const geom_t *const SFEM_RESTRICT x = points[0];
    const geom_t *const SFEM_RESTRICT y = points[1];
    const geom_t *const SFEM_RESTRICT z = points[2];

#pragma omp parallel for
    for (ptrdiff_t e = 0; e < nelements; ++e) {
        idx_t  ev[27];
        real_t element_vector[27] = {0};

        for (int v = 0; v < 27; ++v) {
            ev[v] = elements[v][e];
        }

        for (int qz = 0; qz < 3; ++qz) {
            for (int qy = 0; qy < 3; ++qy) {
                for (int qx = 0; qx < 3; ++qx) {
                    real_t dx_dxi = 0;
                    real_t dx_deta = 0;
                    real_t dx_dzeta = 0;
                    real_t dy_dxi = 0;
                    real_t dy_deta = 0;
                    real_t dy_dzeta = 0;
                    real_t dz_dxi = 0;
                    real_t dz_deta = 0;
                    real_t dz_dzeta = 0;

                    for (int v = 0; v < 27; ++v) {
                        const int cart = local_to_cartesian[v];
                        const int lx   = cart % 3;
                        const int ly   = (cart / 3) % 3;
                        const int lz   = cart / 9;

                        const real_t dN_dxi   = grad[qx][lx] * shape[qy][ly] * shape[qz][lz];
                        const real_t dN_deta  = shape[qx][lx] * grad[qy][ly] * shape[qz][lz];
                        const real_t dN_dzeta = shape[qx][lx] * shape[qy][ly] * grad[qz][lz];
                        const idx_t  node     = ev[v];

                        dx_dxi += dN_dxi * x[node];
                        dx_deta += dN_deta * x[node];
                        dx_dzeta += dN_dzeta * x[node];

                        dy_dxi += dN_dxi * y[node];
                        dy_deta += dN_deta * y[node];
                        dy_dzeta += dN_dzeta * y[node];

                        dz_dxi += dN_dxi * z[node];
                        dz_deta += dN_deta * z[node];
                        dz_dzeta += dN_dzeta * z[node];
                    }

                    const real_t det_j = dx_dxi * (dy_deta * dz_dzeta - dy_dzeta * dz_deta) -
                                         dx_deta * (dy_dxi * dz_dzeta - dy_dzeta * dz_dxi) +
                                         dx_dzeta * (dy_dxi * dz_deta - dy_deta * dz_dxi);
                    const int    cart_q = qx + 3 * (qy + 3 * qz);
                    const int    v_q    = cartesian_to_local[cart_q];
                    element_vector[v_q] += fabs(det_j) * weight[qx] * weight[qy] * weight[qz];
                }
            }
        }

        for (int v = 0; v < 27; ++v) {
#pragma omp atomic update
            values[ev[v]] += element_vector[v];
        }
    }
}

void assemble_mass(const int element_type,
                   const ptrdiff_t nelements,
                   const ptrdiff_t nnodes,
                   idx_t **const SFEM_RESTRICT elems,
                   geom_t **const SFEM_RESTRICT xyz,
                   const count_t *const SFEM_RESTRICT rowptr,
                   const idx_t *const SFEM_RESTRICT colidx,
                   real_t *const SFEM_RESTRICT values) {
    if (!nelements) return;

    switch (element_type) {
        case smesh::TET4: {
            tet4_assemble_mass(nelements, nnodes, elems, xyz, rowptr, colidx, values);
            break;
        }
        case smesh::TET10: {
            tet10_assemble_mass(nelements, nnodes, elems, xyz, rowptr, colidx, values);
            break;
        }
        case smesh::TRI3: {
            tri3_assemble_mass(nelements, nnodes, elems, xyz, rowptr, colidx, values);
            break;
        }
        case smesh::TRISHELL3: {
            trishell3_assemble_mass(nelements, nnodes, elems, xyz, rowptr, colidx, values);
            break;
        }
        case smesh::TRI6: {
            tri6_assemble_mass(nelements, nnodes, elems, xyz, rowptr, colidx, values);
            break;
        }

        default: {
            SFEM_ERROR("assemble_mass not implemented for type %s\n",
                       sfem::type_to_string((smesh::ElemType)element_type));
        }
    }
}

void assemble_lumped_mass(const int element_type,
                          const ptrdiff_t nelements,
                          const ptrdiff_t nnodes,
                          idx_t **const SFEM_RESTRICT elems,
                          geom_t **const SFEM_RESTRICT xyz,
                          real_t *const SFEM_RESTRICT values) {
    if (!nelements) return;

    const smesh::ElemType et = (smesh::ElemType)element_type;
    if (sfem::is_semistructured_type(et)) {
        const int level = smesh::semistructured_level(et);
        affine_sshex8_mass_lumped(level, nelements, 0, elems, xyz, values);
        return;
    }

    switch (element_type) {
        case smesh::TRI3: {
            tri3_assemble_lumped_mass(nelements, nnodes, elems, xyz, values);
            break;
        }
        case smesh::TRISHELL3: {
            trishell3_assemble_lumped_mass(nelements, nnodes, elems, xyz, values);
            break;
        }
        case smesh::TRI6: {
            tri6_assemble_lumped_mass(nelements, nnodes, elems, xyz, values);
            break;
        }
        case smesh::TET4: {
            tet4_assemble_lumped_mass(nelements, nnodes, elems, xyz, values);
            break;
        }
        case smesh::TET10: {
            tet10_assemble_lumped_mass(nelements, nnodes, elems, xyz, values);
            break;
        }
        case smesh::BEAM2: {
            beam2_assemble_lumped_mass(nelements, nnodes, elems, xyz, values);
            break;
        }
        case smesh::HEX8: {
            hex8_assemble_lumped_mass(nelements, nnodes, elems, xyz, 1, values);
            break;
        }
        case smesh::HEX27: {
            hex27_assemble_lobatto_lumped_mass(nelements, nnodes, elems, xyz, false, values);
            break;
        }
        case smesh::PROTEUS_HEX27: {
            hex27_assemble_lobatto_lumped_mass(nelements, nnodes, elems, xyz, true, values);
            break;
        }
        default: {
            SFEM_ERROR("assemble_lumped_mass not implemented for type %s\n",
                       sfem::type_to_string((smesh::ElemType)element_type));
        }
    }
}

void apply_inv_lumped_mass(const int element_type,
                           const ptrdiff_t nelements,
                           const ptrdiff_t nnodes,
                           idx_t **const SFEM_RESTRICT elems,
                           geom_t **const SFEM_RESTRICT xyz,
                           const real_t *const x,
                           real_t *const values) {
    if (!nelements) return;

    switch (element_type) {
        case smesh::TRI3: {
            tri3_apply_inv_lumped_mass(nelements, nnodes, elems, xyz, x, values);
            break;
        }
        case smesh::TRISHELL3: {
            trishell3_apply_inv_lumped_mass(nelements, nnodes, elems, xyz, x, values);
            break;
        }
        case smesh::TRI6: {
            tri6_apply_inv_lumped_mass(nelements, nnodes, elems, xyz, x, values);
            break;
        }
        case smesh::TET4: {
            tet4_apply_inv_lumped_mass(nelements, nnodes, elems, xyz, x, values);
            break;
        }
        case smesh::TET10: {
            tet10_apply_inv_lumped_mass(nelements, nnodes, elems, xyz, x, values);
            break;
        }
        case smesh::BEAM2: {
            beam2_apply_inv_lumped_mass(nelements, nnodes, elems, xyz, x, values);
            break;
        }

        default: {
            SFEM_ERROR("apply_inv_lumped_mass not implemented for type %s\n",
                       sfem::type_to_string((smesh::ElemType)element_type));
        }
    }
}

void apply_mass(const int element_type,
                const ptrdiff_t nelements,
                const ptrdiff_t nnodes,
                idx_t **const SFEM_RESTRICT elems,
                geom_t **const SFEM_RESTRICT xyz,
                const ptrdiff_t stride_x,
                const real_t *const x,
                const ptrdiff_t stride_values,
                real_t *const values) {
    if (!nelements) return;

    switch (element_type) {
        case smesh::BEAM2: {
            beam2_apply_mass(nelements, nnodes, elems, xyz, stride_x, x, stride_values, values);
            break;
        }
        case smesh::TRI3: {
            tri3_apply_mass(nelements, nnodes, elems, xyz, stride_x, x, stride_values, values);
            break;
        }
        case smesh::TRISHELL3: {
            trishell3_apply_mass(nelements, nnodes, elems, xyz, stride_x, x, stride_values, values);
            break;
        }
        case smesh::QUADSHELL4: {
            quadshell4_apply_mass(
                    nelements, nnodes, elems, xyz, stride_x, x, stride_values, values);
            break;
        }
        case smesh::HEX8: {
            hex8_apply_mass(nelements, nnodes, elems, xyz, stride_x, x, stride_values, values);
            break;
        }
            // case smesh::TRI6: {
            //         tri6_apply_mass(nelements, nnodes, elems, xyz, x, values);
            //         break;
            //     }

            // case smesh::TET4: {
            //     tet4_apply_mass(nelements, nnodes, elems, xyz, x, values);
            //     break;
            // }

            // case smesh::TET10: {
            //     tet10_apply_mass(nelements, nnodes, elems, xyz, x, values);
            //     break;
            // }

        default: {
            SFEM_ERROR("apply_mass not implemented for type %s\n",
                       sfem::type_to_string((smesh::ElemType)element_type));
        }
    }
}
