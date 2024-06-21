#include "tet10_laplacian.h"

#include "tet10_laplacian_inline_cpu.h"

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

int tet10_laplacian_assemble_value(const ptrdiff_t nelements,
                                    const ptrdiff_t nnodes,
                                    idx_t **const SFEM_RESTRICT elements,
                                    geom_t **const SFEM_RESTRICT points,
                                    const real_t *const SFEM_RESTRICT u,
                                    real_t *const SFEM_RESTRICT value) {
    SFEM_UNUSED(nnodes);

    const geom_t *const x = points[0];
    const geom_t *const y = points[1];
    const geom_t *const z = points[2];

#pragma omp parallel for
    for (ptrdiff_t i = 0; i < nelements; ++i) {
        idx_t ev[10];
        jacobian_t fff[6];
        scalar_t element_u[10];

#pragma unroll(10)
        for (int v = 0; v < 10; ++v) {
            ev[v] = elements[v][i];
        }

        for (int v = 0; v < 10; ++v) {
            element_u[v] = u[ev[v]];
        }

        // Affine transformation
        tet4_fff(
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

        real_t element_scalar = 0;
        tet10_laplacian_energy_fff(fff, element_u, &element_scalar);

#pragma omp atomic update
        *value += element_scalar;
    }

    return 0;
}

int tet10_laplacian_apply(const ptrdiff_t nelements,
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
        jacobian_t fff[6];
        scalar_t element_u[10];
        accumulator_t element_vector[10] = {0};

#pragma unroll(10)
        for (int v = 0; v < 10; ++v) {
            ev[v] = elements[v][i];
        }

        for (int v = 0; v < 10; ++v) {
            element_u[v] = u[ev[v]];
        }

        // Affine transformation
        tet4_fff(
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

        tet10_laplacian_apply_add_fff(fff, element_u, element_vector);

        for (int edof_i = 0; edof_i < 10; ++edof_i) {
            const idx_t dof_i = ev[edof_i];

#pragma omp atomic update
            values[dof_i] += element_vector[edof_i];
        }
    }

    return 0;
}

int tet10_laplacian_assemble_hessian(const ptrdiff_t nelements,
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
        jacobian_t fff[6];
        accumulator_t element_matrix[10 * 10];

#pragma unroll(10)
        for (int v = 0; v < 10; ++v) {
            ev[v] = elements[v][i];
        }

        // Affine transformation
        tet4_fff(
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

        tet10_laplacian_hessian_fff(fff, element_matrix);
        tet10_local_to_global(ev, element_matrix, rowptr, colidx, values);
    }

    return 0;
}

int tet10_laplacian_diag(const ptrdiff_t nelements,
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
        jacobian_t fff[6];
        accumulator_t element_vector[10];

#pragma unroll(10)
        for (int v = 0; v < 10; ++v) {
            ev[v] = elements[v][i];
        }

        // Affine transformation
        tet4_fff(
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

        tet10_laplacian_diag_fff(fff, element_vector);

        for (int edof_i = 0; edof_i < 10; ++edof_i) {
            const idx_t dof_i = ev[edof_i];

#pragma omp atomic update
            diag[dof_i] += element_vector[edof_i];
        }
    }

    return 0;
}

/////////////

int tet10_laplacian_apply_opt(const ptrdiff_t nelements,
                              idx_t **const SFEM_RESTRICT elements,
                              const jacobian_t *const SFEM_RESTRICT fff_all,
                              const real_t *const SFEM_RESTRICT u,
                              real_t *const SFEM_RESTRICT values) {
#pragma omp parallel for
    for (ptrdiff_t i = 0; i < nelements; ++i) {
        idx_t ev[10];
        scalar_t element_u[10];
        accumulator_t element_vector[10]= {0};

        // Affine transformation
        const jacobian_t *const fff = &fff_all[i * 6];

#pragma unroll(10)
        for (int v = 0; v < 10; ++v) {
            ev[v] = elements[v][i];
        }

        for (int v = 0; v < 10; ++v) {
            element_u[v] = u[ev[v]];
        }

        tet10_laplacian_apply_add_fff(fff, element_u, element_vector);

        for (int edof_i = 0; edof_i < 10; ++edof_i) {
            const idx_t dof_i = ev[edof_i];

#pragma omp atomic update
            values[dof_i] += element_vector[edof_i];
        }
    }

    return 0;
}

int tet10_laplacian_diag_opt(const ptrdiff_t nelements,
                             idx_t **const SFEM_RESTRICT elements,
                             const jacobian_t *const SFEM_RESTRICT fff_all,
                             real_t *const SFEM_RESTRICT diag) {
#pragma omp parallel for
    for (ptrdiff_t i = 0; i < nelements; ++i) {
        idx_t ev[10];
        accumulator_t element_vector[10];

        // Affine transformation
        const jacobian_t *const fff = &fff_all[i * 6];

#pragma unroll(10)
        for (int v = 0; v < 10; ++v) {
            ev[v] = elements[v][i];
        }

        tet10_laplacian_diag_fff(fff, element_vector);

        for (int edof_i = 0; edof_i < 10; ++edof_i) {
            const idx_t dof_i = ev[edof_i];

#pragma omp atomic update
            diag[dof_i] += element_vector[edof_i];
        }
    }

    return 0;
}
