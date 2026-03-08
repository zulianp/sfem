#include "mass.hpp"

#include <assert.h>
#include <math.h>
#include <stdio.h>

#include <mpi.h>

#include "crs_graph.hpp"
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
