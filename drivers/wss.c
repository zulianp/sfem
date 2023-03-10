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

#include "operators/grad_p1.h"

static SFEM_INLINE void wss_mag_kernel(const real_t px0,
                                       const real_t px1,
                                       const real_t px2,
                                       const real_t py0,
                                       const real_t py1,
                                       const real_t py2,
                                       const real_t pz0,
                                       const real_t pz1,
                                       const real_t pz2,
                                       // Data
                                       const real_t *const SFEM_RESTRICT shear,
                                       // Output
                                       real_t *const SFEM_RESTRICT wssmag) {
    //FLOATING POINT OPS!
    //      - Result: ADD + ASSIGNMENT + 3*MUL + POW
    //      - Subexpressions: 14*ADD + 4*DIV + 28*MUL + 3*NEG + 13*POW + 9*SUB
    const real_t x0 = -py0 + py2;
    const real_t x1 = -px0 + px1;
    const real_t x2 = -py0 + py1;
    const real_t x3 = -pz0 + pz1;
    const real_t x4 = pow(pow(x1, 2) + pow(x2, 2) + pow(x3, 2), -1.0/2.0);
    const real_t x5 = -px0 + px2;
    const real_t x6 = -pz0 + pz2;
    const real_t x7 = pow(pow(x0, 2) + pow(x5, 2) + pow(x6, 2), -1.0/2.0);
    const real_t x8 = x4*x7;
    const real_t x9 = x1*x8;
    const real_t x10 = x0*x9 - x2*x5*x8;
    const real_t x11 = pow(x10, 2);
    const real_t x12 = x3*x4*x5*x7 - x6*x9;
    const real_t x13 = pow(x12, 2);
    const real_t x14 = -x0*x3*x8 + x2*x6*x8;
    const real_t x15 = pow(x14, 2);
    const real_t x16 = x11 + x13 + x15;
    const real_t x17 = pow(x16, -1.0/2.0);
    const real_t x18 = x14*x17;
    const real_t x19 = x12*x17;
    const real_t x20 = x10*x17;
    const real_t x21 = pow(x18*(shear[0]*x18 + shear[1]*x19 + shear[2]*x20) + x19*(shear[1]*x18 + shear[3]*x19 + shear[4]*x20) + x20*(shear[2]*x18 + shear[4]*x19 + shear[5]*x20), 2)/x16;
    wssmag[0] = sqrt(x11*x21 + x13*x21 + x15*x21);
}

void wss_mag_3(const ptrdiff_t nelements,
               const ptrdiff_t nnodes,
               idx_t **const elems,
               geom_t **const xyz,
               real_t *const shear_xx,
               real_t *const shear_xy,
               real_t *const shear_xz,
               real_t *const shear_yy,
               real_t *const shear_yz,
               real_t *const shear_zz,
               real_t *const values) {
    SFEM_UNUSED(nnodes);
    
    double tick = MPI_Wtime();

    idx_t ev[4];
    

    for (ptrdiff_t i = 0; i < nelements; ++i) {
#pragma unroll(4)
        for (int v = 0; v < 3; ++v) {
            ev[v] = elems[v][i];
        }

        // Element indices
        const idx_t i0 = ev[0];
        const idx_t i1 = ev[1];
        const idx_t i2 = ev[2];
        const idx_t i3 = ev[3];

        real_t shear[6];
        shear[0] = shear_xx[i];
        shear[1] = shear_xy[i];
        shear[2] = shear_xz[i];
        shear[3] = shear_yy[i];
        shear[4] = shear_yz[i];
        shear[5] = shear_zz[i];

        wss_mag_kernel(
            // X-coordinates
            xyz[0][i0],
            xyz[0][i1],
            xyz[0][i2],

            // Y-coordinates
            xyz[1][i0],
            xyz[1][i1],
            xyz[1][i2],

            // Z-coordinates
            xyz[2][i0],
            xyz[2][i1],
            xyz[2][i2],

            // Data
            shear,
            // Output
            &values[i]);
    }

    double tock = MPI_Wtime();
    printf("wss.c: wss_mag\t%g seconds\n", tock - tick);
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

    if (argc != 4) {
        fprintf(stderr, "usage: %s <folder> <shear_prefix> <wssmag.raw>\n", argv[0]);
        return EXIT_FAILURE;
    }

    const char *folder = argv[1];
    const char *shear_prefix = argv[2];
    const char *path_output = argv[3];

    printf("%s %s %s %s\n", argv[0], folder, shear_prefix, path_output);

    double tick = MPI_Wtime();

    ///////////////////////////////////////////////////////////////////////////////
    // Read data
    ///////////////////////////////////////////////////////////////////////////////

    mesh_t mesh;
    if (mesh_surf_read(comm, folder, &mesh)) {
        return EXIT_FAILURE;
    }

    real_t *shear_6[6];
    ptrdiff_t shear_size_local, shear_size_global;

    char path[2048];

    // P0
    for (int d = 0; d < 6; ++d) {
        sprintf(path, "%s.%d.raw", shear_prefix, d);
        array_create_from_file(comm, path, SFEM_MPI_REAL_T, (void **)&shear_6[d], &shear_size_local, &shear_size_global);
    }

    // P0
    real_t *wss_mag = (real_t *)malloc(mesh.nelements * sizeof(real_t));
    memset(wss_mag, 0, mesh.nelements * sizeof(real_t));

    wss_mag_3(mesh.nelements,
              mesh.nnodes,
              mesh.elements,
              mesh.points,
              shear_6[0],
              shear_6[1],
              shear_6[2],
              shear_6[3],
              shear_6[4],
              shear_6[5],
              wss_mag);

    array_write(comm, path_output, SFEM_MPI_REAL_T, wss_mag, mesh.nelements, mesh.nelements);

    for (int d = 0; d < 6; ++d) {
        free(shear_6[d]);
    }

    free(wss_mag);

    double tock = MPI_Wtime();

    if (!rank) {
        printf("----------------------------------------\n");
        printf("#elements %ld #nodes %ld\n", (long)mesh.nelements, (long)mesh.nnodes);
        printf("TTS:\t\t\t%g seconds\n", tock - tick);
    }

    return MPI_Finalize();
}
