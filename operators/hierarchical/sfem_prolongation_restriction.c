#include "sfem_prolongation_restriction.h"

#include "sfem_base.h"
#include "sfem_defs.h"

#include <mpi.h>
#include <stdio.h>

int hierarchical_prolongation(const enum ElemType from_element,
                              const enum ElemType to_element,
                              const ptrdiff_t nelements,
                              idx_t **const SFEM_RESTRICT elements,
                              const real_t *const SFEM_RESTRICT from,
                              real_t *const SFEM_RESTRICT to) {
    if (from_element == TET4 && (to_element == TET10 || to_element == MACRO_TET4)) {
#pragma omp parallel for
        for (ptrdiff_t e = 0; e < nelements; e++) {
            // P1
            const idx_t i0 = elements[e][0];
            const idx_t i1 = elements[e][1];
            const idx_t i2 = elements[e][2];
            const idx_t i3 = elements[e][3];

            // P2
            const idx_t i4 = elements[e][4];
            const idx_t i5 = elements[e][5];
            const idx_t i6 = elements[e][6];
            const idx_t i7 = elements[e][7];
            const idx_t i8 = elements[e][8];
            const idx_t i9 = elements[e][9];

            to[i0] = from[i0];
            to[i1] = from[i1];
            to[i2] = from[i2];
            to[i3] = from[i3];

            to[i4] = 0.5 * (from[i0] + from[i1]);
            to[i5] = 0.5 * (from[i1] + from[i2]);
            to[i6] = 0.5 * (from[i0] + from[i2]);
            to[i7] = 0.5 * (from[i0] + from[i3]);
            to[i8] = 0.5 * (from[i1] + from[i3]);
            to[i9] = 0.5 * (from[i2] + from[i3]);
        }
    } else if (from_element == TRI3 && (to_element == TRI6 || to_element == MACRO_TRI3)) {
#pragma omp parallel for
        for (ptrdiff_t e = 0; e < nelements; e++) {
            // P1
            const idx_t i0 = elements[e][0];
            const idx_t i1 = elements[e][1];
            const idx_t i2 = elements[e][2];

            // P2
            const idx_t i3 = elements[e][3];
            const idx_t i4 = elements[e][4];
            const idx_t i5 = elements[e][5];

            to[i0] = from[i0];
            to[i1] = from[i1];
            to[i2] = from[i2];

            to[i3] = 0.5 * (from[i0] + from[i1]);
            to[i4] = 0.5 * (from[i1] + from[i2]);
            to[i5] = 0.5 * (from[i0] + from[i2]);
        }
    } else {
        assert(0);
        fprintf(stderr,
                "Unsupported element pair for hierarchical_prolongation %d, %d\n",
                from_element,
                to_element);
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }

    return 0;
}

int hierarchical_restriction(const enum ElemType from_element,
                             const enum ElemType to_element,
                             const ptrdiff_t nelements,
                             idx_t **const SFEM_RESTRICT elements,
                             const real_t *const SFEM_RESTRICT from,
                             real_t *const SFEM_RESTRICT to) {
    if (to_element == TET4 && (from_element == TET10 || from_element == MACRO_TET4)) {
#pragma omp parallel for
        for (ptrdiff_t e = 0; e < nelements; e++) {
            // P1
            const idx_t i0 = elements[e][0];
            const idx_t i1 = elements[e][1];
            const idx_t i2 = elements[e][2];
            const idx_t i3 = elements[e][3];

            // P2
            const idx_t i4 = elements[e][4];
            const idx_t i5 = elements[e][5];
            const idx_t i6 = elements[e][6];
            const idx_t i7 = elements[e][7];
            const idx_t i8 = elements[e][8];
            const idx_t i9 = elements[e][9];

            to[i0] = from[i0] + 0.5 * (from[i4] + from[i6] + from[i7]);
            to[i1] = from[i1] + 0.5 * (from[i4] + from[i5] + from[i8]);
            to[i2] = from[i2] + 0.5 * (from[i5] + from[i6] + from[i9]);
            to[i3] = from[i3] + 0.5 * (from[i7] + from[i8] + from[i9]);
        }

    } else if (to_element == TRI3 && (from_element == TRI6 || from_element == MACRO_TRI3)) {
#pragma omp parallel for
        for (ptrdiff_t e = 0; e < nelements; e++) {
            // P1
            const idx_t i0 = elements[e][0];
            const idx_t i1 = elements[e][1];
            const idx_t i2 = elements[e][2];

            // P2
            const idx_t i3 = elements[e][3];
            const idx_t i4 = elements[e][4];
            const idx_t i5 = elements[e][5];

            to[i0] = from[i0] + 0.5 * (from[i3] + from[i5]);
            to[i1] = from[i1] + 0.5 * (from[i3] + from[i4]);
            to[i2] = from[i2] + 0.5 * (from[i4] + from[i5]);
        }
    } else {
        assert(0);
        fprintf(stderr,
                "Unsupported element pair for hierarchical_restriction %d, %d\n",
                from_element,
                to_element);
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }

    return 0;
}
