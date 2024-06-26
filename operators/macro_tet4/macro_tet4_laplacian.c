#include "macro_tet4_laplacian.h"

#include "sfem_base.h"

#include "tet4_inline_cpu.h"
#include "tet4_laplacian_inline_cpu.h"
#include "macro_tet4_inline_cpu.h"

#include <assert.h>
#include <stdio.h>


static SFEM_INLINE void element_apply(const idx_t *const SFEM_RESTRICT ev,
                                      const scalar_t *const SFEM_RESTRICT fff,
                                      const scalar_t *const SFEM_RESTRICT element_u,
                                      real_t *const SFEM_RESTRICT values) {
    scalar_t sub_fff[6];
    accumulator_t element_vector[10] = {0};

    {  // Corner tests
        tet4_sub_fff_0(fff, sub_fff);

        // [0, 4, 6, 7],
        tet4_laplacian_apply_add_fff(sub_fff,
                                     element_u[0],
                                     element_u[4],
                                     element_u[6],
                                     element_u[7],
                                     &element_vector[0],
                                     &element_vector[4],
                                     &element_vector[6],
                                     &element_vector[7]);

        // [4, 1, 5, 8],
        tet4_laplacian_apply_add_fff(sub_fff,
                                     element_u[4],
                                     element_u[1],
                                     element_u[5],
                                     element_u[8],
                                     &element_vector[4],
                                     &element_vector[1],
                                     &element_vector[5],
                                     &element_vector[8]);

        // [6, 5, 2, 9],
        tet4_laplacian_apply_add_fff(sub_fff,
                                     element_u[6],
                                     element_u[5],
                                     element_u[2],
                                     element_u[9],
                                     &element_vector[6],
                                     &element_vector[5],
                                     &element_vector[2],
                                     &element_vector[9]);

        // [7, 8, 9, 3],
        tet4_laplacian_apply_add_fff(sub_fff,
                                     element_u[7],
                                     element_u[8],
                                     element_u[9],
                                     element_u[3],
                                     &element_vector[7],
                                     &element_vector[8],
                                     &element_vector[9],
                                     &element_vector[3]);
    }

    {  // Octahedron tets

        // [4, 5, 6, 8],
        tet4_sub_fff_4(fff, sub_fff);
        tet4_laplacian_apply_add_fff(sub_fff,
                                     element_u[4],
                                     element_u[5],
                                     element_u[6],
                                     element_u[8],
                                     &element_vector[4],
                                     &element_vector[5],
                                     &element_vector[6],
                                     &element_vector[8]);

        // [7, 4, 6, 8],
        tet4_sub_fff_5(fff, sub_fff);
        tet4_laplacian_apply_add_fff(sub_fff,
                                     element_u[7],
                                     element_u[4],
                                     element_u[6],
                                     element_u[8],
                                     &element_vector[7],
                                     &element_vector[4],
                                     &element_vector[6],
                                     &element_vector[8]);

        // [6, 5, 9, 8],
        tet4_sub_fff_6(fff, sub_fff);
        tet4_laplacian_apply_add_fff(sub_fff,
                                     element_u[6],
                                     element_u[5],
                                     element_u[9],
                                     element_u[8],
                                     &element_vector[6],
                                     &element_vector[5],
                                     &element_vector[9],
                                     &element_vector[8]);

        // [7, 6, 9, 8]]
        tet4_sub_fff_7(fff, sub_fff);
        tet4_laplacian_apply_add_fff(sub_fff,
                                     element_u[7],
                                     element_u[6],
                                     element_u[9],
                                     element_u[8],
                                     &element_vector[7],
                                     &element_vector[6],
                                     &element_vector[9],
                                     &element_vector[8]);
    }

#pragma unroll(10)
    for (int v = 0; v < 10; v++) {
#pragma omp atomic update
        values[ev[v]] += element_vector[v];
    }
}

int macro_tet4_laplacian_apply(const ptrdiff_t nelements,
                               const ptrdiff_t nnodes,
                               idx_t **const SFEM_RESTRICT elements,
                               geom_t **const SFEM_RESTRICT points,
                               const real_t *const SFEM_RESTRICT u,
                               real_t *const SFEM_RESTRICT values) {
    SFEM_UNUSED(nnodes);

    const geom_t *const x = points[0];
    const geom_t *const y = points[1];
    const geom_t *const z = points[2];

#pragma omp parallel for
    for (ptrdiff_t i = 0; i < nelements; ++i) {
        idx_t ev[10];
        scalar_t fff[6];
        scalar_t element_u[10];

        // Element indices
#pragma unroll(10)
        for (int v = 0; v < 10; ++v) {
            ev[v] = elements[v][i];
        }

        for (int v = 0; v < 10; ++v) {
            element_u[v] = u[ev[v]];
        }

        tet4_fff_s(
                // X
                x[ev[0]],
                x[ev[1]],
                x[ev[2]],
                x[ev[3]],
                // Y
                y[ev[0]],
                y[ev[1]],
                y[ev[2]],
                y[ev[3]],
                // Z
                z[ev[0]],
                z[ev[1]],
                z[ev[2]],
                z[ev[3]],
                //
                fff);

        element_apply(ev, fff, element_u, values);
    }

    return 0;
}

int macro_tet4_laplacian_apply_opt(const ptrdiff_t nelements,
                                   idx_t **const SFEM_RESTRICT elements,
                                   const jacobian_t *const SFEM_RESTRICT fff_all,
                                   const real_t *const SFEM_RESTRICT u,
                                   real_t *const SFEM_RESTRICT values) {
#pragma omp parallel for  // nowait
    for (ptrdiff_t i = 0; i < nelements; ++i) {
        idx_t ev[10];
        scalar_t element_u[10];
        scalar_t fff[6];

        for(int k = 0; k < 6; k++) {
            fff[k] = fff_all[i * 6 + k];
        }


#pragma unroll(10)
        for (int v = 0; v < 10; ++v) {
            ev[v] = elements[v][i];
        }

        for (int v = 0; v < 10; ++v) {
            element_u[v] = u[ev[v]];
        }

        element_apply(ev, fff, element_u, values);
    }

    return 0;
}

static SFEM_INLINE void element_assemble_matrix(const idx_t *const SFEM_RESTRICT ev10,
                                                const scalar_t *const fff,
                                                const count_t *const SFEM_RESTRICT rowptr,
                                                const idx_t *const SFEM_RESTRICT colidx,
                                                real_t *const SFEM_RESTRICT values) {
    idx_t ev[4];
    accumulator_t element_matrix[4 * 4];
    scalar_t sub_fff[6];

    {  // Corner tests
        tet4_sub_fff_0(fff, sub_fff);
        tet4_laplacian_hessian_fff(sub_fff, element_matrix);

        // [0, 4, 6, 7],
        tet4_gather_idx(ev10, 0, 4, 6, 7, ev);
        tet4_local_to_global(ev, element_matrix, rowptr, colidx, values);

        // [4, 1, 5, 8],
        tet4_gather_idx(ev10, 4, 1, 5, 8, ev);
        tet4_local_to_global(ev, element_matrix, rowptr, colidx, values);

        // [6, 5, 2, 9],
        tet4_gather_idx(ev10, 6, 5, 2, 9, ev);
        tet4_local_to_global(ev, element_matrix, rowptr, colidx, values);

        // [7, 8, 9, 3],
        tet4_gather_idx(ev10, 7, 8, 9, 3, ev);
        tet4_local_to_global(ev, element_matrix, rowptr, colidx, values);
    }

    {  // Octahedron tets
        // [4, 5, 6, 8],
        tet4_sub_fff_4(fff, sub_fff);
        tet4_laplacian_hessian_fff(sub_fff, element_matrix);
        tet4_gather_idx(ev10, 4, 5, 6, 8, ev);
        tet4_local_to_global(ev, element_matrix, rowptr, colidx, values);

        // [7, 4, 6, 8],
        tet4_sub_fff_5(fff, sub_fff);
        tet4_laplacian_hessian_fff(sub_fff, element_matrix);
        tet4_gather_idx(ev10, 7, 4, 6, 8, ev);
        tet4_local_to_global(ev, element_matrix, rowptr, colidx, values);

        // [6, 5, 9, 8],
        tet4_sub_fff_6(fff, sub_fff);
        tet4_laplacian_hessian_fff(sub_fff, element_matrix);
        tet4_gather_idx(ev10, 6, 5, 9, 8, ev);
        tet4_local_to_global(ev, element_matrix, rowptr, colidx, values);

        // [7, 6, 9, 8]]
        tet4_sub_fff_7(fff, sub_fff);
        tet4_laplacian_hessian_fff(sub_fff, element_matrix);
        tet4_gather_idx(ev10, 7, 6, 9, 8, ev);
        tet4_local_to_global(ev, element_matrix, rowptr, colidx, values);
    }
}

int macro_tet4_laplacian_assemble_hessian_opt(const ptrdiff_t nelements,
                                              idx_t **const SFEM_RESTRICT elements,
                                              const jacobian_t *const SFEM_RESTRICT fff_all,
                                              const count_t *const SFEM_RESTRICT rowptr,
                                              const idx_t *const SFEM_RESTRICT colidx,
                                              real_t *const SFEM_RESTRICT values) {
#pragma omp parallel for
    for (ptrdiff_t i = 0; i < nelements; ++i) {
        idx_t ev10[10];
        scalar_t fff[6];

        for(int k = 0; k < 6; k++) {
            fff[k] = fff_all[i * 6 + k];
        }

        for (int v = 0; v < 10; ++v) {
            ev10[v] = elements[v][i];
        }

        element_assemble_matrix(ev10, fff, rowptr, colidx, values);
    }

    return 0;
}

int macro_tet4_laplacian_assemble_hessian(const ptrdiff_t nelements,
                                          const ptrdiff_t nnodes,
                                          idx_t **const SFEM_RESTRICT elements,
                                          geom_t **const SFEM_RESTRICT points,
                                          const count_t *const SFEM_RESTRICT rowptr,
                                          const idx_t *const SFEM_RESTRICT colidx,
                                          real_t *const SFEM_RESTRICT values) {
    SFEM_UNUSED(nnodes);

    const geom_t *const x = points[0];
    const geom_t *const y = points[1];
    const geom_t *const z = points[2];

#pragma omp parallel for
    for (ptrdiff_t i = 0; i < nelements; ++i) {
        idx_t ev[10];
        scalar_t fff[6];

        // Element indices
        for (int v = 0; v < 10; ++v) {
            ev[v] = elements[v][i];
        }

        tet4_fff_s(
                // X
                x[ev[0]],
                x[ev[1]],
                x[ev[2]],
                x[ev[3]],
                // Y
                y[ev[0]],
                y[ev[1]],
                y[ev[2]],
                y[ev[3]],
                // Z
                z[ev[0]],
                z[ev[1]],
                z[ev[2]],
                z[ev[3]],
                //
                fff);

        element_assemble_matrix(ev, fff, rowptr, colidx, values);
    }

    return 0;
}

static SFEM_INLINE void element_assemble_matrix_diag(const idx_t *const SFEM_RESTRICT ev,
                                                     const scalar_t *const fff,
                                                     real_t *const SFEM_RESTRICT diag) {
    scalar_t sub_fff[6];
    accumulator_t element_vector[10] = {0};

    {  // Corner tests
        tet4_sub_fff_0(fff, sub_fff);

        // [0, 4, 6, 7],
        tet4_laplacian_diag_add_fff(sub_fff,
                                    &element_vector[0],
                                    &element_vector[4],
                                    &element_vector[6],
                                    &element_vector[7]);

        // [4, 1, 5, 8],
        tet4_laplacian_diag_add_fff(sub_fff,
                                    &element_vector[4],
                                    &element_vector[1],
                                    &element_vector[5],
                                    &element_vector[8]);

        // [6, 5, 2, 9],
        tet4_laplacian_diag_add_fff(sub_fff,
                                    &element_vector[6],
                                    &element_vector[5],
                                    &element_vector[2],
                                    &element_vector[9]);

        // [7, 8, 9, 3],
        tet4_laplacian_diag_add_fff(sub_fff,
                                    &element_vector[7],
                                    &element_vector[8],
                                    &element_vector[9],
                                    &element_vector[3]);
    }

    {  // Octahedron tets

        // [4, 5, 6, 8],
        tet4_sub_fff_4(fff, sub_fff);
        tet4_laplacian_diag_add_fff(sub_fff,
                                    &element_vector[4],
                                    &element_vector[5],
                                    &element_vector[6],
                                    &element_vector[8]);

        // [7, 4, 6, 8],
        tet4_sub_fff_5(fff, sub_fff);
        tet4_laplacian_diag_add_fff(sub_fff,
                                    &element_vector[7],
                                    &element_vector[4],
                                    &element_vector[6],
                                    &element_vector[8]);

        // [6, 5, 9, 8],
        tet4_sub_fff_6(fff, sub_fff);
        tet4_laplacian_diag_add_fff(sub_fff,
                                    &element_vector[6],
                                    &element_vector[5],
                                    &element_vector[9],
                                    &element_vector[8]);

        // [7, 6, 9, 8]]
        tet4_sub_fff_7(fff, sub_fff);
        tet4_laplacian_diag_add_fff(sub_fff,
                                    &element_vector[7],
                                    &element_vector[6],
                                    &element_vector[9],
                                    &element_vector[8]);
    }

#pragma unroll(10)
    for (int v = 0; v < 10; v++) {
#pragma omp atomic update
        diag[ev[v]] += element_vector[v];
    }
}

int macro_tet4_laplacian_diag(const ptrdiff_t nelements,
                              const ptrdiff_t nnodes,
                              idx_t **const SFEM_RESTRICT elements,
                              geom_t **const SFEM_RESTRICT points,
                              real_t *const SFEM_RESTRICT diag) {
    SFEM_UNUSED(nnodes);

    const geom_t *const x = points[0];
    const geom_t *const y = points[1];
    const geom_t *const z = points[2];

#pragma omp parallel for
    for (ptrdiff_t i = 0; i < nelements; ++i) {
        idx_t ev[10];
        scalar_t fff[6];

        // Element indices
        for (int v = 0; v < 10; ++v) {
            ev[v] = elements[v][i];
        }

        tet4_fff_s(
                // X
                x[ev[0]],
                x[ev[1]],
                x[ev[2]],
                x[ev[3]],
                // Y
                y[ev[0]],
                y[ev[1]],
                y[ev[2]],
                y[ev[3]],
                // Z
                z[ev[0]],
                z[ev[1]],
                z[ev[2]],
                z[ev[3]],
                //
                fff);

        element_assemble_matrix_diag(ev, fff, diag);
    }

    return 0;
}

int macro_tet4_laplacian_diag_opt(const ptrdiff_t nelements,
                                  idx_t **const SFEM_RESTRICT elements,
                                  const jacobian_t *const SFEM_RESTRICT fff_all,
                                  real_t *const SFEM_RESTRICT diag) {
#pragma omp parallel for
    for (ptrdiff_t i = 0; i < nelements; ++i) {
        idx_t ev[10];

        scalar_t fff[6];
        for(int k = 0; k < 6; k++) {
            fff[k] = fff_all[i * 6 + k];
        }

        assert(tet4_det_fff(fff) > 0);

#pragma unroll(10)
        for (int v = 0; v < 10; ++v) {
            ev[v] = elements[v][i];
        }

        element_assemble_matrix_diag(ev, fff, diag);
    }

    return 0;
}
