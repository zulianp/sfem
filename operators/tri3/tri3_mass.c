#include "tri3_mass.h"

#include <mpi.h>
#include <stdio.h>

#include "crs_graph.h"
#include "sortreduce.h"

#include "sfem_vec.h"

static SFEM_INLINE void tri3_apply_mass_kernel(const real_t px0,
                                               const real_t px1,
                                               const real_t px2,
                                               const real_t py0,
                                               const real_t py1,
                                               const real_t py2,
                                               const real_t *const SFEM_RESTRICT u,
                                               real_t *const SFEM_RESTRICT element_vector) {
    const real_t x0 =
        (1.0 / 24.0) * (px0 - px1) * (py0 - py2) - 1.0 / 24.0 * (px0 - px2) * (py0 - py1);
    element_vector[0] = x0 * (2 * u[0] + u[1] + u[2]);
    element_vector[1] = x0 * (u[0] + 2 * u[1] + u[2]);
    element_vector[2] = x0 * (u[0] + u[1] + 2 * u[2]);
}

void tri3_apply_mass(const ptrdiff_t nelements,
                     const ptrdiff_t nnodes,
                     idx_t **const SFEM_RESTRICT elems,
                     geom_t **const SFEM_RESTRICT xyz,
                     const real_t *const x,
                     real_t *const values) {
    SFEM_UNUSED(nnodes);

    double tick = MPI_Wtime();

#pragma omp parallel
    {
#pragma omp for //nowait
        for (ptrdiff_t i = 0; i < nelements; ++i) {
            idx_t ev[3];
            idx_t ks[3];
            real_t element_x[3];
            real_t element_vector[3];

#pragma unroll(3)
            for (int v = 0; v < 3; ++v) {
                ev[v] = elems[v][i];
            }

            // Element indices
            const idx_t i0 = ev[0];
            const idx_t i1 = ev[1];
            const idx_t i2 = ev[2];

            for (int enode = 0; enode < 3; ++enode) {
                element_x[enode] = x[ev[enode]];
            }

            tri3_apply_mass_kernel(  
                // X-coordinates
                xyz[0][i0],
                xyz[0][i1],
                xyz[0][i2],
                // Y-coordinates
                xyz[1][i0],
                xyz[1][i1],
                xyz[1][i2],
                element_x,
                // output vector
                element_vector);

#pragma unroll(3)
            for (int edof_i = 0; edof_i < 3; edof_i++) {
#pragma omp atomic update
                values[ev[edof_i]] += element_vector[edof_i];
            }
        }
    }

    double tock = MPI_Wtime();
    printf("tri3_mass.c: tri3_apply_mass\t%g seconds\n", tock - tick);
}
