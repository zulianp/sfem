#include "tet4_laplacian.h"
#include "tet4_inline_cpu.h"
#include "tet4_laplacian_inline_cpu.h"

#include <assert.h>
#include <math.h>
#include <mpi.h>
#include <stdio.h>

// Classic mesh-based assembly

void tet4_laplacian_assemble_value(const ptrdiff_t nelements,
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
        real_t element_u[4];

        // Element indices
#pragma unroll(4)
        for (int v = 0; v < 4; ++v) {
            ev[v] = elements[v][i];
        }

        for (int v = 0; v < 4; ++v) {
            element_u[v] = u[ev[v]];
        }

        real_t element_scalar = 0;
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
                x[ev[0]],
                x[ev[1]],
                x[ev[2]],
                x[ev[3]],
                element_u,
                &element_scalar);

#pragma omp atomic update
        *value += element_scalar;
    }
}

void tet4_laplacian_apply(const ptrdiff_t nelements,
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
        idx_t ev[4];
        real_t element_vector[4];
        real_t element_u[4];

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
}

void tet4_laplacian_assemble_gradient(const ptrdiff_t nelements,
                                      const ptrdiff_t nnodes,
                                      idx_t **const SFEM_RESTRICT elements,
                                      geom_t **const SFEM_RESTRICT points,
                                      const real_t *const SFEM_RESTRICT u,
                                      real_t *const SFEM_RESTRICT values) {
    tet4_laplacian_apply(nelements, nnodes, elements, points, u, values);
}

void tet4_laplacian_assemble_hessian(const ptrdiff_t nelements,
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
        real_t element_matrix[4 * 4];

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
}

void tet4_laplacian_diag(const ptrdiff_t nelements,
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
        real_t element_vector[4];

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
}

// Optimized for matrix-free
int tet4_laplacian_apply_opt(const ptrdiff_t nelements,
                             idx_t **const SFEM_RESTRICT elements,
                             const jacobian_t *const SFEM_RESTRICT fff,
                             const real_t *const SFEM_RESTRICT u,
                             real_t *const SFEM_RESTRICT values) {
#pragma omp parallel for
    for (ptrdiff_t i = 0; i < nelements; ++i) {
        real_t element_vector[4];
        idx_t ev[4];

        // Element indices
#pragma unroll(4)
        for (int v = 0; v < 4; ++v) {
            ev[v] = elements[v][i];
        }

        tet4_laplacian_apply_fff(&fff[i * 6],
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
                            const jacobian_t *const SFEM_RESTRICT fff,
                            real_t *const SFEM_RESTRICT diag) {
#pragma omp parallel for
    for (ptrdiff_t i = 0; i < nelements; ++i) {
        idx_t ev[4];
        real_t element_vector[4];

        // Element indices
#pragma unroll(4)
        for (int v = 0; v < 4; ++v) {
            ev[v] = elements[v][i];
        }

        tet4_laplacian_diag_fff(&fff[i * 6],
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
