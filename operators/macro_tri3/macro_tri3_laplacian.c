#include "macro_tri3_laplacian.h"

#include <assert.h>
#include <stdio.h>

#include "sfem_base.h"
#include "sortreduce.h"

#include "tri3_inline_cpu.h"
#include "tri3_laplacian_inline_cpu.h"

#include "macro_tri3_inline_cpu.h"

static SFEM_INLINE void element_apply(const idx_t *const SFEM_RESTRICT ev,
                                      const scalar_t *const SFEM_RESTRICT fff,
                                      const scalar_t *const SFEM_RESTRICT element_u,
                                      accumulator_t *const SFEM_RESTRICT values) {
    scalar_t sub_fff[3];
    accumulator_t element_vector[6] = {0};

    {
        // Corner FFFs (Same as fff)
        tri3_laplacian_apply_add_fff(fff,
                                     element_u[0],
                                     element_u[3],
                                     element_u[5],
                                     &element_vector[0],
                                     &element_vector[3],
                                     &element_vector[5]);

        tri3_laplacian_apply_add_fff(fff,
                                     element_u[3],
                                     element_u[1],
                                     element_u[4],
                                     &element_vector[3],
                                     &element_vector[1],
                                     &element_vector[4]);

        tri3_laplacian_apply_add_fff(fff,
                                     element_u[5],
                                     element_u[4],
                                     element_u[2],
                                     &element_vector[5],
                                     &element_vector[4],
                                     &element_vector[2]);
    }

    {  // Central FFF
        tri3_sub_fff_1(fff, sub_fff);
        tri3_laplacian_apply_add_fff(sub_fff,
                                     element_u[3],
                                     element_u[4],
                                     element_u[5],
                                     &element_vector[3],
                                     &element_vector[4],
                                     &element_vector[5]);
    }

#pragma unroll(6)
    for (int v = 0; v < 6; v++) {
#pragma omp atomic update
        values[ev[v]] += element_vector[v];
    }
}

int macro_tri3_laplacian_apply(const ptrdiff_t nelements,
                               const ptrdiff_t nnodes,
                               idx_t **const SFEM_RESTRICT elements,
                               geom_t **const SFEM_RESTRICT points,
                               const real_t *const SFEM_RESTRICT u,
                               real_t *const SFEM_RESTRICT values) {
    SFEM_UNUSED(nnodes);

    const geom_t *const x = points[0];
    const geom_t *const y = points[1];

#pragma omp parallel for
    for (ptrdiff_t i = 0; i < nelements; ++i) {
        idx_t ev[6];
        scalar_t fff[3];
        scalar_t element_u[6];

        // Element indices
#pragma unroll(6)
        for (int v = 0; v < 6; ++v) {
            ev[v] = elements[v][i];
        }

        for (int v = 0; v < 6; ++v) {
            element_u[v] = u[ev[v]];
        }

        tri3_fff_s(
                // X
                x[ev[0]],
                x[ev[1]],
                x[ev[2]],
                // Y
                y[ev[0]],
                y[ev[1]],
                y[ev[2]],
                //
                fff);

        element_apply(ev, fff, element_u, values);
    }

    return 0;
}

int macro_tri3_laplacian_apply_opt(const ptrdiff_t nelements,
                                   idx_t **const SFEM_RESTRICT elements,
                                   const jacobian_t *const SFEM_RESTRICT fff_all,
                                   const real_t *const SFEM_RESTRICT u,
                                   real_t *const SFEM_RESTRICT values) {
#pragma omp parallel for  // nowait
    for (ptrdiff_t i = 0; i < nelements; ++i) {
        idx_t ev[6];
        scalar_t element_u[6];
        scalar_t fff[3];
        
        for(int k = 0; k < 3; k++) {
            fff[k] = fff_all[i * 3 + k];
        }


#pragma unroll(6)
        for (int v = 0; v < 6; ++v) {
            ev[v] = elements[v][i];
        }

        for (int v = 0; v < 6; ++v) {
            element_u[v] = u[ev[v]];
        }

        element_apply(ev, fff, element_u, values);
    }

    return 0;
}

static SFEM_INLINE void element_assemble_matrix(const idx_t *const SFEM_RESTRICT ev6,
                                                const scalar_t *const fff,
                                                const count_t *const SFEM_RESTRICT rowptr,
                                                const idx_t *const SFEM_RESTRICT colidx,
                                                accumulator_t *const SFEM_RESTRICT values) {
    idx_t ev[3];
    accumulator_t element_matrix[3 * 3];
    scalar_t sub_fff[3];

    {
        // Corner FFFs (Same as fff)
        tri3_laplacian_hessian_fff(fff, element_matrix);

        tri3_gather_idx(ev6, 0, 3, 5, ev);
        tri3_local_to_global(ev, element_matrix, rowptr, colidx, values);

        tri3_gather_idx(ev6, 3, 1, 4, ev);
        tri3_local_to_global(ev, element_matrix, rowptr, colidx, values);

        tri3_gather_idx(ev6, 5, 4, 2, ev);
        tri3_local_to_global(ev, element_matrix, rowptr, colidx, values);
    }

    {  // Central FFF
        tri3_sub_fff_1(fff, sub_fff);
        tri3_laplacian_hessian_fff(sub_fff, element_matrix);
        tri3_gather_idx(ev6, 3, 4, 5, ev);
        tri3_local_to_global(ev, element_matrix, rowptr, colidx, values);
    }
}

int macro_tri3_laplacian_assemble_hessian_opt(const ptrdiff_t nelements,
                                              idx_t **const SFEM_RESTRICT elements,
                                              const jacobian_t *const SFEM_RESTRICT fff_all,
                                              const count_t *const SFEM_RESTRICT rowptr,
                                              const idx_t *const SFEM_RESTRICT colidx,
                                              real_t *const SFEM_RESTRICT values) {
#pragma omp parallel for
    for (ptrdiff_t i = 0; i < nelements; ++i) {
        idx_t ev6[6];

        scalar_t fff[3];
        for(int k = 0; k < 3; k++) {
            fff[k] = fff_all[i * 3 + k];
        }

        for (int v = 0; v < 6; ++v) {
            ev6[v] = elements[v][i];
        }

        element_assemble_matrix(ev6, fff, rowptr, colidx, values);
    }

    return 0;
}

int macro_tri3_laplacian_assemble_hessian(const ptrdiff_t nelements,
                                          const ptrdiff_t nnodes,
                                          idx_t **const SFEM_RESTRICT elements,
                                          geom_t **const SFEM_RESTRICT points,
                                          const count_t *const SFEM_RESTRICT rowptr,
                                          const idx_t *const SFEM_RESTRICT colidx,
                                          real_t *const SFEM_RESTRICT values) {
    SFEM_UNUSED(nnodes);

    const geom_t *const x = points[0];
    const geom_t *const y = points[1];

#pragma omp parallel for
    for (ptrdiff_t i = 0; i < nelements; ++i) {
        idx_t ev[6];
        scalar_t fff[3];

        // Element indices
        for (int v = 0; v < 6; ++v) {
            ev[v] = elements[v][i];
        }

        tri3_fff_s(
                // X
                x[ev[0]],
                x[ev[1]],
                x[ev[2]],
                // Y
                y[ev[0]],
                y[ev[1]],
                y[ev[2]],
                //
                fff);

        element_assemble_matrix(ev, fff, rowptr, colidx, values);
    }

    return 0;
}

static SFEM_INLINE void element_assemble_matrix_diag(const idx_t *const SFEM_RESTRICT ev,
                                                     const scalar_t *const fff,
                                                     real_t *const SFEM_RESTRICT diag) {
    scalar_t sub_fff[3];
    accumulator_t element_vector[6] = {0};

    {
        // Corner FFFs (Same as fff)
        tri3_laplacian_diag_add_fff(
                fff, &element_vector[0], &element_vector[3], &element_vector[5]);

        tri3_laplacian_diag_add_fff(
                fff, &element_vector[3], &element_vector[1], &element_vector[4]);

        tri3_laplacian_diag_add_fff(
                fff, &element_vector[5], &element_vector[4], &element_vector[2]);
    }

    {  // Central FFF
        tri3_sub_fff_1(fff, sub_fff);
        tri3_laplacian_diag_add_fff(
                sub_fff, &element_vector[3], &element_vector[4], &element_vector[5]);
    }

#pragma unroll(6)
    for (int v = 0; v < 6; v++) {
#pragma omp atomic update
        diag[ev[v]] += element_vector[v];
    }
}

int macro_tri3_laplacian_diag(const ptrdiff_t nelements,
                              const ptrdiff_t nnodes,
                              idx_t **const SFEM_RESTRICT elements,
                              geom_t **const SFEM_RESTRICT points,
                              real_t *const SFEM_RESTRICT diag) {
    SFEM_UNUSED(nnodes);

    const geom_t *const x = points[0];
    const geom_t *const y = points[1];

#pragma omp parallel for
    for (ptrdiff_t i = 0; i < nelements; ++i) {
        idx_t ev[6];
        scalar_t fff[3];

        // Element indices
        for (int v = 0; v < 6; ++v) {
            ev[v] = elements[v][i];
        }

        tri3_fff_s(
                // X
                x[ev[0]],
                x[ev[1]],
                x[ev[2]],
                // Y
                y[ev[0]],
                y[ev[1]],
                y[ev[2]],
                //
                fff);

        element_assemble_matrix_diag(ev, fff, diag);
    }

    return 0;
}

int macro_tri3_laplacian_diag_opt(const ptrdiff_t nelements,
                                  idx_t **const SFEM_RESTRICT elements,
                                  const jacobian_t *const SFEM_RESTRICT fff_all,
                                  real_t *const SFEM_RESTRICT diag) {
#pragma omp parallel for
    for (ptrdiff_t i = 0; i < nelements; ++i) {
        idx_t ev[6];
        scalar_t fff[3];
        for(int k = 0; k < 3; k++) {
            fff[k] = fff_all[i * 3 + k];
        }

        assert(tri3_det_fff(fff) > 0);

#pragma unroll(6)
        for (int v = 0; v < 6; ++v) {
            ev[v] = elements[v][i];
        }

        element_assemble_matrix_diag(ev, fff, diag);
    }

    return 0;
}
