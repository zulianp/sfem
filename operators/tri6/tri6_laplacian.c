#include "tri6_laplacian.h"

#include "tri6_laplacian_inline_cpu.h"

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

int tri6_laplacian_assemble_value(const ptrdiff_t nelements,
                                  const ptrdiff_t nnodes,
                                  idx_t **const SFEM_RESTRICT elements,
                                  geom_t **const SFEM_RESTRICT points,
                                  const real_t *const SFEM_RESTRICT u,
                                  real_t *const SFEM_RESTRICT value) {
    SFEM_UNUSED(nnodes);

    const geom_t *const x = points[0];
    const geom_t *const y = points[1];

#pragma omp parallel for
    for (ptrdiff_t i = 0; i < nelements; ++i) {
        idx_t ev[6];
        jacobian_t fff[6];
        scalar_t element_u[6];

#pragma unroll(6)
        for (int v = 0; v < 6; ++v) {
            ev[v] = elements[v][i];
        }

        for (int v = 0; v < 6; ++v) {
            element_u[v] = u[ev[v]];
        }

        // Affine transformation
        tri3_fff(
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

        real_t element_scalar = 0;
        tri6_laplacian_energy_fff(fff, element_u, &element_scalar);

#pragma omp atomic update
        *value += element_scalar;
    }

    return 0;
}

int tri6_laplacian_apply(const ptrdiff_t nelements,
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
        jacobian_t fff[6];
        scalar_t element_u[6];
        accumulator_t element_vector[6] = {0};

#pragma unroll(6)
        for (int v = 0; v < 6; ++v) {
            ev[v] = elements[v][i];
        }

        for (int v = 0; v < 6; ++v) {
            element_u[v] = u[ev[v]];
        }

        // Affine transformation
        tri3_fff(
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

        tri6_laplacian_apply_fff(fff, element_u, element_vector);

        for (int edof_i = 0; edof_i < 6; ++edof_i) {
            const idx_t dof_i = ev[edof_i];

#pragma omp atomic update
            values[dof_i] += element_vector[edof_i];
        }
    }

    return 0;
}

int tri6_laplacian_crs(const ptrdiff_t nelements,
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
        jacobian_t fff[6];
        accumulator_t element_matrix[6 * 6];

#pragma unroll(6)
        for (int v = 0; v < 6; ++v) {
            ev[v] = elements[v][i];
        }

        // Affine transformation
        tri3_fff(
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

        tri6_laplacian_hessian_fff(fff, element_matrix);
        tri6_local_to_global(ev, element_matrix, rowptr, colidx, values);
    }

    return 0;
}

int tri6_laplacian_diag(const ptrdiff_t nelements,
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
        jacobian_t fff[6];
        accumulator_t element_vector[6];

#pragma unroll(6)
        for (int v = 0; v < 6; ++v) {
            ev[v] = elements[v][i];
        }

        // Affine transformation
        tri3_fff(
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

        tri6_laplacian_diag_fff(fff, element_vector);

        for (int edof_i = 0; edof_i < 6; ++edof_i) {
            const idx_t dof_i = ev[edof_i];

#pragma omp atomic update
            diag[dof_i] += element_vector[edof_i];
        }
    }

    return 0;
}

/////////////

int tri6_laplacian_apply_opt(const ptrdiff_t nelements,
                             idx_t **const SFEM_RESTRICT elements,
                             const jacobian_t *const SFEM_RESTRICT fff_all,
                             const real_t *const SFEM_RESTRICT u,
                             real_t *const SFEM_RESTRICT values) {
#pragma omp parallel for
    for (ptrdiff_t i = 0; i < nelements; ++i) {
        idx_t ev[6];
        scalar_t element_u[6];
        accumulator_t element_vector[6];

        // Affine transformation
        const jacobian_t *const fff = &fff_all[i * 3];

#pragma unroll(6)
        for (int v = 0; v < 6; ++v) {
            ev[v] = elements[v][i];
        }

        for (int v = 0; v < 6; ++v) {
            element_u[v] = u[ev[v]];
        }

        tri6_laplacian_apply_fff(fff, element_u, element_vector);

        for (int edof_i = 0; edof_i < 6; ++edof_i) {
            const idx_t dof_i = ev[edof_i];

#pragma omp atomic update
            values[dof_i] += element_vector[edof_i];
        }
    }

    return 0;
}

int tri6_laplacian_diag_opt(const ptrdiff_t nelements,
                            idx_t **const SFEM_RESTRICT elements,
                            const jacobian_t *const SFEM_RESTRICT fff_all,
                            real_t *const SFEM_RESTRICT diag) {
#pragma omp parallel for
    for (ptrdiff_t i = 0; i < nelements; ++i) {
        idx_t ev[6];
        accumulator_t element_vector[6];

        // Affine transformation
        const jacobian_t *const fff = &fff_all[i * 3];

#pragma unroll(6)
        for (int v = 0; v < 6; ++v) {
            ev[v] = elements[v][i];
        }

        tri6_laplacian_diag_fff(fff, element_vector);

        for (int edof_i = 0; edof_i < 6; ++edof_i) {
            const idx_t dof_i = ev[edof_i];

#pragma omp atomic update
            diag[dof_i] += element_vector[edof_i];
        }
    }

    return 0;
}
