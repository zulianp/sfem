#include "neumann.h"

#include <assert.h>
#include <math.h>
#include <mpi.h>
#include <stdio.h>

static SFEM_INLINE real_t det3(const real_t *mat) {
    return mat[0] * mat[4] * mat[8] + mat[1] * mat[5] * mat[6] + mat[2] * mat[3] * mat[7] - mat[0] * mat[5] * mat[7] -
           mat[1] * mat[3] * mat[8] - mat[2] * mat[4] * mat[6];
}

SFEM_INLINE real_t area3(const real_t left[3], const real_t right[3]) {
    real_t a = (left[1] * right[2]) - (right[1] * left[2]);
    real_t b = (left[2] * right[0]) - (right[2] * left[0]);
    real_t c = (left[0] * right[1]) - (right[0] * left[1]);
    return sqrt(a * a + b * b + c * c);
}

void surface_forcing_function(const ptrdiff_t nfaces,
                              const idx_t *faces_neumann,
                              geom_t **const xyz,
                              const real_t value,
                              real_t *output) {  // Neumann
    double tick = MPI_Wtime();

    for (idx_t f = 0; f < nfaces; ++f) {
        idx_t i0 = faces_neumann[f * 3];
        idx_t i1 = faces_neumann[f * 3 + 1];
        idx_t i2 = faces_neumann[f * 3 + 2];

        real_t u[3], v[3];

        for (int d = 0; d < 3; d++) {
            real_t x0 = (real_t)xyz[d][i0];
            real_t x1 = (real_t)xyz[d][i1];
            real_t x2 = (real_t)xyz[d][i2];
            u[d] = x1 - x0;
            v[d] = x2 - x0;
        }

        real_t dx = area3(u, v) / 2;

        assert(dx > 0.);

        real_t integr = value * dx;

        output[i0] += integr;
        output[i1] += integr;
        output[i2] += integr;
    }

    double tock = MPI_Wtime();
    printf("neumann.c: surface_forcing_function\t%g seconds\n", tock - tick);
}

void surface_forcing_function_vec(const ptrdiff_t nfaces,
                                  const idx_t *faces_neumann,
                                  geom_t **const xyz,
                                  const real_t value,
                                  const int block_size,
                                  const int component,
                                  real_t *output) {
    double tick = MPI_Wtime();

    real_t jacobian[3 * 3] = {0, 0, 0, 0, 0, 0, 0, 0, 1};

    // real_t value = 2.0;
    for (idx_t f = 0; f < nfaces; ++f) {
        idx_t i0 = faces_neumann[f * 3];
        idx_t i1 = faces_neumann[f * 3 + 1];
        idx_t i2 = faces_neumann[f * 3 + 2];

        // No square roots in this version
        for (int d = 0; d < 3; ++d) {
            real_t x0 = (real_t)xyz[d][i0];
            real_t x1 = (real_t)xyz[d][i1];
            real_t x2 = (real_t)xyz[d][i2];

            jacobian[d * 3] = x1 - x0;
            jacobian[d * 3 + 1] = x2 - x0;
        }

        // Orientation of face is not proper
        real_t dx = fabs(det3(jacobian)) / 2;

        assert(dx > 0.);

        real_t integr = value * dx;

        output[i0 * block_size + component] += integr;
        output[i1 * block_size + component] += integr;
        output[i2 * block_size + component] += integr;
    }

    double tock = MPI_Wtime();
    printf("neumann.c: surface_forcing_function_vec\t%g seconds\n", tock - tick);
}
