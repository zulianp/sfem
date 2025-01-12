#include "tet4_laplacian.h"

#include "tet4_inline_cpu.h"
#include "tet4_laplacian_inline_cpu.h"

#ifdef SFEM_ENABLE_EXPLICIT_VECTORIZATION
#include "vtet4_laplacian.h"
#endif

#include <assert.h>
#include <math.h>
#include <stdio.h>

// Classic mesh-based assembly

int tet4_laplacian_assemble_value(const ptrdiff_t nelements,
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
        idx_t ev[4];
        scalar_t element_u[4];

        // Element indices
#pragma unroll(4)
        for (int v = 0; v < 4; ++v) {
            ev[v] = elements[v][i];
        }

        for (int v = 0; v < 4; ++v) {
            element_u[v] = u[ev[v]];
        }

        accumulator_t element_scalar = 0;
        tet4_laplacian_value_points(
                // X-coordinates
                x[ev[0]],
                x[ev[1]],
                x[ev[2]],
                x[ev[3]],
                // Y-coordinates
                y[ev[0]],
                y[ev[1]],
                y[ev[2]],
                y[ev[3]],
                // Z-coordinates
                z[ev[0]],
                z[ev[1]],
                z[ev[2]],
                z[ev[3]],
                element_u,
                &element_scalar);

#pragma omp atomic update
        *value += element_scalar;
    }

    return 0;
}

int tet4_laplacian_apply(const ptrdiff_t nelements,
                         const ptrdiff_t nnodes,
                         idx_t **const SFEM_RESTRICT elements,
                         geom_t **const SFEM_RESTRICT points,
                         const real_t *const SFEM_RESTRICT u,
                         real_t *const SFEM_RESTRICT values) {
#ifdef SFEM_ENABLE_EXPLICIT_VECTORIZATION
    int SFEM_ENABLE_V = 0;
    SFEM_READ_ENV(SFEM_ENABLE_V, atoi);

    if (SFEM_ENABLE_V) {
        return vtet4_laplacian_apply(nelements, nnodes, elements, points, u, values);
    }
#endif

    SFEM_UNUSED(nnodes);

    const geom_t *const x = points[0];
    const geom_t *const y = points[1];
    const geom_t *const z = points[2];

#pragma omp parallel for
    for (ptrdiff_t i = 0; i < nelements; ++i) {
        idx_t ev[4];
        accumulator_t element_vector[4];
        scalar_t element_u[4];

        // Element indices
#pragma unroll(4)
        for (int v = 0; v < 4; ++v) {
            ev[v] = elements[v][i];
        }

        for (int v = 0; v < 4; ++v) {
            element_u[v] = u[ev[v]];
        }

        tet4_laplacian_apply_points(
                // X-coordinates
                x[ev[0]],
                x[ev[1]],
                x[ev[2]],
                x[ev[3]],
                // Y-coordinates
                y[ev[0]],
                y[ev[1]],
                y[ev[2]],
                y[ev[3]],
                // Z-coordinates
                z[ev[0]],
                z[ev[1]],
                z[ev[2]],
                z[ev[3]],
                element_u,
                element_vector);

        for (int edof_i = 0; edof_i < 4; ++edof_i) {
            const idx_t dof_i = ev[edof_i];

#pragma omp atomic update
            values[dof_i] += element_vector[edof_i];
        }
    }

    return 0;
}

int tet4_laplacian_crs(const ptrdiff_t nelements,
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
        idx_t ev[4];
        accumulator_t element_matrix[4 * 4];

        // Element indices
#pragma unroll(4)
        for (int v = 0; v < 4; ++v) {
            ev[v] = elements[v][i];
        }

        tet4_laplacian_hessian_points(
                // X-coordinates
                x[ev[0]],
                x[ev[1]],
                x[ev[2]],
                x[ev[3]],
                // Y-coordinates
                y[ev[0]],
                y[ev[1]],
                y[ev[2]],
                y[ev[3]],
                // Z-coordinates
                z[ev[0]],
                z[ev[1]],
                z[ev[2]],
                z[ev[3]],
                element_matrix);

        tet4_local_to_global(ev, element_matrix, rowptr, colidx, values);
    }

    return 0;
}

int tet4_laplacian_diag(const ptrdiff_t nelements,
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
        idx_t ev[4];
        accumulator_t element_vector[4];

        // Element indices
#pragma unroll(4)
        for (int v = 0; v < 4; ++v) {
            ev[v] = elements[v][i];
        }

        tet4_laplacian_diag_points(
                // X-coordinates
                x[ev[0]],
                x[ev[1]],
                x[ev[2]],
                x[ev[3]],
                // Y-coordinates
                y[ev[0]],
                y[ev[1]],
                y[ev[2]],
                y[ev[3]],
                // Z-coordinates
                z[ev[0]],
                z[ev[1]],
                z[ev[2]],
                z[ev[3]],
                element_vector);

        for (int edof_i = 0; edof_i < 4; ++edof_i) {
            const idx_t dof_i = ev[edof_i];

#pragma omp atomic update
            diag[dof_i] += element_vector[edof_i];
        }
    }

    return 0;
}

// Optimized for matrix-free
int tet4_laplacian_apply_opt(const ptrdiff_t nelements,
                             idx_t **const SFEM_RESTRICT elements,
                             const jacobian_t *const SFEM_RESTRICT fff_all,
                             const real_t *const SFEM_RESTRICT u,
                             real_t *const SFEM_RESTRICT values) {
#ifdef SFEM_ENABLE_EXPLICIT_VECTORIZATION
    int SFEM_ENABLE_V = 0;
    SFEM_READ_ENV(SFEM_ENABLE_V, atoi);

    if (SFEM_ENABLE_V) {
        return vtet4_laplacian_apply_opt(nelements, elements, fff_all, u, values);
    }
#endif

#pragma omp parallel for
    for (ptrdiff_t i = 0; i < nelements; ++i) {
        accumulator_t element_vector[4];
        idx_t ev[4];
        scalar_t fff[6];
        for (int k = 0; k < 6; k++) {
            fff[k] = fff_all[i * 6 + k];
        }

        // Element indices
#pragma unroll(4)
        for (int v = 0; v < 4; ++v) {
            ev[v] = elements[v][i];
        }

        tet4_laplacian_apply_fff(fff,
                                 u[ev[0]],
                                 u[ev[1]],
                                 u[ev[2]],
                                 u[ev[3]],
                                 &element_vector[0],
                                 &element_vector[1],
                                 &element_vector[2],
                                 &element_vector[3]);

        for (int edof_i = 0; edof_i < 4; ++edof_i) {
            const idx_t dof_i = ev[edof_i];

#pragma omp atomic update
            values[dof_i] += element_vector[edof_i];
        }
    }

    return 0;
}

int tet4_laplacian_diag_opt(const ptrdiff_t nelements,
                            idx_t **const SFEM_RESTRICT elements,
                            const jacobian_t *const SFEM_RESTRICT fff_all,
                            real_t *const SFEM_RESTRICT diag) {
#pragma omp parallel for
    for (ptrdiff_t i = 0; i < nelements; ++i) {
        idx_t ev[4];
        accumulator_t element_vector[4];

        scalar_t fff[6];
        for (int k = 0; k < 6; k++) {
            fff[k] = fff_all[i * 6 + k];
        }

        // Element indices
#pragma unroll(4)
        for (int v = 0; v < 4; ++v) {
            ev[v] = elements[v][i];
        }

        tet4_laplacian_diag_fff(fff,
                                &element_vector[0],
                                &element_vector[1],
                                &element_vector[2],
                                &element_vector[3]);

        for (int edof_i = 0; edof_i < 4; ++edof_i) {
            const idx_t dof_i = ev[edof_i];

#pragma omp atomic update
            diag[dof_i] += element_vector[edof_i];
        }
    }

    return 0;
}
