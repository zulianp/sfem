#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "array_dtof.h"
#include "matrixio_array.h"
#include "matrixio_crs.h"
#include "utils.h"

#include "crs_graph.h"
#include "sfem_base.h"

#include "read_mesh.h"

// Microkernel
static SFEM_INLINE void cgradient(const real_t px0,
                                  const real_t px1,
                                  const real_t px2,
                                  const real_t px3,
                                  const real_t py0,
                                  const real_t py1,
                                  const real_t py2,
                                  const real_t py3,
                                  const real_t pz0,
                                  const real_t pz1,
                                  const real_t pz2,
                                  const real_t pz3,
                                  // Data
                                  const real_t *SFEM_RESTRICT f,
                                  // Output
                                  real_t *SFEM_RESTRICT dfdx,
                                  real_t *SFEM_RESTRICT dfdy,
                                  real_t *SFEM_RESTRICT dfdz) {
    //FLOATING POINT OPS!
    //      - Result: 6*ADD + 3*ASSIGNMENT + 21*MUL
    //      - Subexpressions: 2*ADD + DIV + 34*MUL + 21*SUB
    const real_t x0 = -py0 + py2;
    const real_t x1 = -pz0 + pz3;
    const real_t x2 = x0*x1;
    const real_t x3 = -py0 + py3;
    const real_t x4 = -pz0 + pz2;
    const real_t x5 = x3*x4;
    const real_t x6 = -px0 + px1;
    const real_t x7 = -px0 + px2;
    const real_t x8 = -pz0 + pz1;
    const real_t x9 = -px0 + px3;
    const real_t x10 = -py0 + py1;
    const real_t x11 = x10*x4;
    const real_t x12 = x1*x10;
    const real_t x13 = x0*x8;
    const real_t x14 = 1.0/(x11*x9 - x12*x7 - x13*x9 + x2*x6 + x3*x7*x8 - x5*x6);
    const real_t x15 = x14*(x2 - x5);
    const real_t x16 = x14*(-x12 + x3*x8);
    const real_t x17 = x14*(x11 - x13);
    const real_t x18 = x14*(-x1*x7 + x4*x9);
    const real_t x19 = x14*(x1*x6 - x8*x9);
    const real_t x20 = x14*(-x4*x6 + x7*x8);
    const real_t x21 = x14*(-x0*x9 + x3*x7);
    const real_t x22 = x14*(x10*x9 - x3*x6);
    const real_t x23 = x14*(x0*x6 - x10*x7);
    dfdx[0] = f[0]*(-x15 - x16 - x17) + f[1]*x15 + f[2]*x16 + f[3]*x17;
    dfdy[0] = f[0]*(-x18 - x19 - x20) + f[1]*x18 + f[2]*x19 + f[3]*x20;
    dfdz[0] = f[0]*(-x21 - x22 - x23) + f[1]*x21 + f[2]*x22 + f[3]*x23;
}

void cgrad3(const ptrdiff_t nelements,
            const ptrdiff_t nnodes,
            idx_t **const SFEM_RESTRICT elems,
            geom_t **SFEM_RESTRICT xyz,
            const real_t *const SFEM_RESTRICT f,
            real_t *const SFEM_RESTRICT dfdx,
            real_t *const SFEM_RESTRICT dfdy,
            real_t *const SFEM_RESTRICT dfdz) {
    SFEM_UNUSED(nnodes);

    double tick = MPI_Wtime();

    idx_t ev[4];
    real_t element_f[4];
    real_t element_dfdx;
    real_t element_dfdy;
    real_t element_dfdz;

    for (ptrdiff_t i = 0; i < nelements; ++i) {
#pragma unroll(4)
        for (int v = 0; v < 4; ++v) {
            ev[v] = elems[v][i];
        }

        for (int v = 0; v < 4; ++v) {
            element_f[v] = f[ev[v]];
        }

        // Element indices
        const idx_t i0 = ev[0];
        const idx_t i1 = ev[1];
        const idx_t i2 = ev[2];
        const idx_t i3 = ev[3];

        cgradient(
            // X-coordinates
            xyz[0][i0],
            xyz[0][i1],
            xyz[0][i2],
            xyz[0][i3],
            // Y-coordinates
            xyz[1][i0],
            xyz[1][i1],
            xyz[1][i2],
            xyz[1][i3],
            // Z-coordinates
            xyz[2][i0],
            xyz[2][i1],
            xyz[2][i2],
            xyz[2][i3],
            // Data
            element_f,
            // Output
            &element_dfdx,
            &element_dfdy,
            &element_dfdz);

        // Write cell data
        dfdx[i] = element_dfdx;
        dfdy[i] = element_dfdy;
        dfdz[i] = element_dfdz;
    }

    double tock = MPI_Wtime();
    printf("cgrad.c: cgrad3\t%g seconds\n", tock - tick);
}

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    MPI_Comm comm = MPI_COMM_WORLD;

    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    if (size != 1) {
        fprintf(stderr, "Parallel execution not supported!\n");
        return EXIT_FAILURE;
    }

    if (argc < 6) {
        fprintf(stderr, "usage: %s <folder> <f.raw> <dfdx.raw> <dfdy.raw> <dfdz.raw>\n", argv[0]);
        return EXIT_FAILURE;
    }

    const char *folder = argv[1];
    const char *path_f = argv[2];
    const char *path_outputs[3] = {argv[3], argv[4], argv[5]};

    printf("%s %s %s %s %s %s\n", argv[0], folder, path_f, path_outputs[0], path_outputs[1], path_outputs[2]);

    double tick = MPI_Wtime();

    ///////////////////////////////////////////////////////////////////////////////
    // Read data
    ///////////////////////////////////////////////////////////////////////////////

    mesh_t mesh;
    if (mesh_read(comm, folder, &mesh)) {
        return EXIT_FAILURE;
    }

    real_t *f;
    ptrdiff_t u_n_local, u_n_global;
    array_read(comm, path_f, SFEM_MPI_REAL_T, (void **)&f, &u_n_local, &u_n_global);

    ptrdiff_t nelements = mesh.nelements;

    real_t *df[3];
    for (int d = 0; d < 3; ++d) {
        df[d] = (real_t *)malloc(nelements * sizeof(real_t));
        memset(df[d], 0, nelements * sizeof(real_t));
    }

    ///////////////////////////////////////////////////////////////////////////////
    // Compute gradient coefficients
    ///////////////////////////////////////////////////////////////////////////////

    cgrad3(mesh.nelements, mesh.nnodes, mesh.elements, mesh.points, f, df[0], df[1], df[2]);

    ///////////////////////////////////////////////////////////////////////////////
    // Write cell data
    ///////////////////////////////////////////////////////////////////////////////

    for (int d = 0; d < 3; ++d) {
        array_write(comm, path_outputs[d], SFEM_MPI_REAL_T, df[d], nelements, nelements);
    }

    ///////////////////////////////////////////////////////////////////////////////
    // Free resources
    ///////////////////////////////////////////////////////////////////////////////

    free(f);
    for (int d = 0; d < 3; ++d) {
        free(df[d]);
    }

    double tock = MPI_Wtime();

    if (!rank) {
        printf("----------------------------------------\n");
        printf("#elements %ld #nodes %ld\n", (long)mesh.nelements, (long)mesh.nnodes);
        printf("TTS:\t\t\t%g seconds\n", tock - tick);
    }

    return MPI_Finalize();
}
