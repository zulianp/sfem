#include "neumann.h"

#include <assert.h>
#include <math.h>
#include <mpi.h>
#include <stdio.h>

static SFEM_INLINE real_t det3(const real_t *mat) {
    return mat[0] * mat[4] * mat[8] + mat[1] * mat[5] * mat[6] + mat[2] * mat[3] * mat[7] - mat[0] * mat[5] * mat[7] -
           mat[1] * mat[3] * mat[8] - mat[2] * mat[4] * mat[6];
}

void surface_forcing_function(const ptrdiff_t nfaces,
                              const idx_t *faces_neumann,
                              geom_t **const xyz,
                              const real_t value,
                              real_t *output) {  // Neumann
    double tick = MPI_Wtime();

    real_t u[3], v[3];
    real_t element_vector[3];
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

        output[i0] += integr;
        output[i1] += integr;
        output[i2] += integr;
    }

    double tock = MPI_Wtime();
    printf("laplacian.c: laplacian_assemble_value\t%g seconds\n", tock - tick);
}
