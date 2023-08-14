#include "tri6_mass.h"

#include <assert.h>
#include <math.h>
#include <stddef.h>
#include <stdio.h>
#include <string.h>

#include <mpi.h>

#include "crs_graph.h"
#include "sortreduce.h"

#include "sfem_base.h"
#include "sfem_vec.h"


static SFEM_INLINE void lumped_mass_kernel(const real_t px0,
                                           const real_t px1,
                                           const real_t px2,
                                           const real_t py0,
                                           const real_t py1,
                                           const real_t py2,
                                           real_t *const SFEM_RESTRICT element_matrix_diag) {
	const real_t x0 = (px0 - px1)*(py0 - py2);
	const real_t x1 = (px0 - px2)*(py0 - py1);
	const real_t x2 = (1.0/15.0)*x0 - 1.0/15.0*x1;
	const real_t x3 = (1.0/10.0)*x0 - 1.0/10.0*x1;
	element_matrix_diag[0] = x2;
	element_matrix_diag[1] = x2;
	element_matrix_diag[2] = x2;
	element_matrix_diag[3] = x3;
	element_matrix_diag[4] = x3;
	element_matrix_diag[5] = x3;
}

static SFEM_INLINE void tri6_transform_kernel(const real_t *const SFEM_RESTRICT x,
                                               real_t *const SFEM_RESTRICT values) {
	const real_t x0 = (1.0/5.0)*x[0];
	const real_t x1 = (1.0/5.0)*x[1];
	const real_t x2 = (1.0/5.0)*x[2];
	values[0] = x[0];
	values[1] = x[1];
	values[2] = x[2];
	values[3] = x0 + x1 + (3.0/5.0)*x[3];
	values[4] = x1 + x2 + (3.0/5.0)*x[4];
	values[5] = x0 + x2 + (3.0/5.0)*x[5];
}

void tri6_apply_inv_lumped_mass(const ptrdiff_t nelements,
                                  const ptrdiff_t nnodes,
                                  idx_t **const SFEM_RESTRICT elems,
                                  geom_t **const SFEM_RESTRICT xyz,
                                  const real_t *const x,
                                  real_t *const values) {
    double tick = MPI_Wtime();

    idx_t ev[6];
    real_t element_x[6];
    real_t element_x_pre_trafo[6];
    real_t element_weights[6];

    // Apply diagonal
    {
        real_t *weights = (real_t *)malloc(nnodes * sizeof(real_t));
        memset(weights, 0, nnodes * sizeof(real_t));

        for (ptrdiff_t i = 0; i < nelements; ++i) {
#pragma unroll(6)
            for (int v = 0; v < 6; ++v) {
                ev[v] = elems[v][i];
            }

            // Element indices
            const idx_t i0 = ev[0];
            const idx_t i1 = ev[1];
            const idx_t i2 = ev[2];


            lumped_mass_kernel(
                // X-coordinates
                xyz[0][i0],
                xyz[0][i1],
                xyz[0][i2],
                // Y-coordinates
                xyz[1][i0],
                xyz[1][i1],
                xyz[1][i2],
                element_weights);

            for (int v = 0; v < 6; ++v) {
                const idx_t idx = ev[v];
                weights[idx] += element_weights[v];
            }
        }

        for (ptrdiff_t i = 0; i < nnodes; i++) {
            values[i] = x[i] / weights[i];
        }

        free(weights);
    }

    for (ptrdiff_t i = 0; i < nelements; ++i) {
#pragma unroll(6)
        for (int v = 0; v < 6; ++v) {
            element_x_pre_trafo[v] = values[elems[v][i]];
        }

        tri6_transform_kernel(element_x_pre_trafo, element_x);

        for (int v = 0; v < 6; ++v) {
            const idx_t idx = ev[v];
            values[idx] = element_x[v];
        }
    }

    double tock = MPI_Wtime();
    printf("tri6_mass.c: tri6_apply_inv_lumped_mass\t%g seconds\n", tock - tick);
}
