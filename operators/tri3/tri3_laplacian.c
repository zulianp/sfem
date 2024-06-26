#include "tri3_laplacian.h"

#include "tri3_inline_cpu.h"
#include "tri3_laplacian_inline_cpu.h"

#include <assert.h>
#include <math.h>
#include <stdio.h>

// Classic mesh-based assembly

int tri3_laplacian_assemble_value(const ptrdiff_t nelements,
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
        idx_t ev[3];
        scalar_t element_u[3];
        scalar_t fff[3];

        // Element indices
#pragma unroll(3)
        for (int v = 0; v < 3; ++v) {
            ev[v] = elements[v][i];
        }

        tri3_fff_s(
                // X-coordinates
                x[ev[0]],
                x[ev[1]],
                x[ev[2]],
                // Y-coordinates
                y[ev[0]],
                y[ev[1]],
                y[ev[2]],
                fff);

        accumulator_t element_scalar = 0;
        tri3_laplacian_energy_fff(fff, u[ev[0]], u[ev[1]], u[ev[2]], &element_scalar);

#pragma omp atomic update
        *value += element_scalar;
    }

    return 0;
}

int tri3_laplacian_apply(const ptrdiff_t nelements,
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
        idx_t ev[3];
        scalar_t fff[3];
        scalar_t element_u[3];
        accumulator_t element_vector[3];
        
        // Element indices
#pragma unroll(3)
        for (int v = 0; v < 3; ++v) {
            ev[v] = elements[v][i];
        }

        tri3_fff_s(
                // X-coordinates
                x[ev[0]],
                x[ev[1]],
                x[ev[2]],
                // Y-coordinates
                y[ev[0]],
                y[ev[1]],
                y[ev[2]],
                fff);

        tri3_laplacian_apply_fff(fff,
                                 u[ev[0]],
                                 u[ev[1]],
                                 u[ev[2]],
                                 &element_vector[0],
                                 &element_vector[1],
                                 &element_vector[2]);

        for (int edof_i = 0; edof_i < 3; ++edof_i) {
            const idx_t dof_i = ev[edof_i];

#pragma omp atomic update
            values[dof_i] += element_vector[edof_i];
        }
    }

    return 0;
}

int tri3_laplacian_assemble_hessian(const ptrdiff_t nelements,
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
        idx_t ev[3];
        scalar_t fff[3];
        accumulator_t element_matrix[3 * 3];

        // Element indices
#pragma unroll(3)
        for (int v = 0; v < 3; ++v) {
            ev[v] = elements[v][i];
        }

        tri3_fff_s(
                // X-coordinates
                x[ev[0]],
                x[ev[1]],
                x[ev[2]],
                // Y-coordinates
                y[ev[0]],
                y[ev[1]],
                y[ev[2]],
                fff);

        tri3_laplacian_hessian_fff(
                fff,
                element_matrix);

        tri3_local_to_global(ev, element_matrix, rowptr, colidx, values);
    }

    return 0;
}

int tri3_laplacian_diag(const ptrdiff_t nelements,
                        const ptrdiff_t nnodes,
                        idx_t **const SFEM_RESTRICT elements,
                        geom_t **const SFEM_RESTRICT points,
                        real_t *const SFEM_RESTRICT diag) {
    SFEM_UNUSED(nnodes);

    const geom_t *const x = points[0];
    const geom_t *const y = points[1];

#pragma omp parallel for
    for (ptrdiff_t i = 0; i < nelements; ++i) {
        idx_t ev[3];
        scalar_t fff[3];
        accumulator_t element_vector[3];

        // Element indices
#pragma unroll(3)
        for (int v = 0; v < 3; ++v) {
            ev[v] = elements[v][i];
        }

        tri3_fff_s(
                // X-coordinates
                x[ev[0]],
                x[ev[1]],
                x[ev[2]],
                // Y-coordinates
                y[ev[0]],
                y[ev[1]],
                y[ev[2]],
                fff);

        tri3_laplacian_diag_fff(
                fff,
                // Output
                &element_vector[0],
                &element_vector[1],
                &element_vector[2]);

        for (int edof_i = 0; edof_i < 3; ++edof_i) {
            const idx_t dof_i = ev[edof_i];

#pragma omp atomic update
            diag[dof_i] += element_vector[edof_i];
        }
    }

    return 0;
}

// Optimized for matrix-free
int tri3_laplacian_apply_opt(const ptrdiff_t nelements,
                             idx_t **const SFEM_RESTRICT elements,
                             const jacobian_t *const SFEM_RESTRICT fff_all,
                             const real_t *const SFEM_RESTRICT u,
                             real_t *const SFEM_RESTRICT values) {
#pragma omp parallel for
    for (ptrdiff_t i = 0; i < nelements; ++i) {
        accumulator_t element_vector[3];
        idx_t ev[3];

        scalar_t fff[3];
        for(int k = 0; k < 3; k++) {
            fff[k] = fff_all[i * 3 + k];
        }


        // Element indices
#pragma unroll(3)
        for (int v = 0; v < 3; ++v) {
            ev[v] = elements[v][i];
        }

        tri3_laplacian_apply_fff(&fff[i * 3],
                                 u[ev[0]],
                                 u[ev[1]],
                                 u[ev[2]],
                                 &element_vector[0],
                                 &element_vector[1],
                                 &element_vector[2]
                                 );

        for (int edof_i = 0; edof_i < 3; ++edof_i) {
            const idx_t dof_i = ev[edof_i];

#pragma omp atomic update
            values[dof_i] += element_vector[edof_i];
        }
    }

    return 0;
}

int tri3_laplacian_diag_opt(const ptrdiff_t nelements,
                            idx_t **const SFEM_RESTRICT elements,
                            const jacobian_t *const SFEM_RESTRICT fff_all,
                            real_t *const SFEM_RESTRICT diag) {
#pragma omp parallel for
    for (ptrdiff_t i = 0; i < nelements; ++i) {
        idx_t ev[3];
        accumulator_t element_vector[3];

        scalar_t fff[3];
        for(int k = 0; k < 3; k++) {
            fff[k] = fff_all[i * 3 + k];
        }

        // Element indices
#pragma unroll(3)
        for (int v = 0; v < 3; ++v) {
            ev[v] = elements[v][i];
        }

        tri3_laplacian_diag_fff(&fff[i * 3],
                                &element_vector[0],
                                &element_vector[1],
                                &element_vector[2]);

        for (int edof_i = 0; edof_i < 3; ++edof_i) {
            const idx_t dof_i = ev[edof_i];

#pragma omp atomic update
            diag[dof_i] += element_vector[edof_i];
        }
    }

    return 0;
}
