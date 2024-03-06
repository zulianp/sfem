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

#include "tet4_grad.h"

static SFEM_INLINE void shear_kernel(const real_t px0,
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
                                     const real_t *const SFEM_RESTRICT ux,
                                     const real_t *const SFEM_RESTRICT uy,
                                     const real_t *const SFEM_RESTRICT uz,
                                     // Output
                                     real_t *const SFEM_RESTRICT shear) {
    // FLOATING POINT OPS!
    //       - Result: 6*ADD + 6*ASSIGNMENT + 36*MUL
    //       - Subexpressions: 2*ADD + DIV + 42*MUL + 3*NEG + 27*SUB
    const real_t x0 = -py0 + py2;
    const real_t x1 = -pz0 + pz3;
    const real_t x2 = x0 * x1;
    const real_t x3 = -py0 + py3;
    const real_t x4 = -pz0 + pz2;
    const real_t x5 = x3 * x4;
    const real_t x6 = -px0 + px1;
    const real_t x7 = -px0 + px2;
    const real_t x8 = -pz0 + pz1;
    const real_t x9 = -px0 + px3;
    const real_t x10 = -py0 + py1;
    const real_t x11 = x10 * x4;
    const real_t x12 = x1 * x10;
    const real_t x13 = x0 * x8;
    const real_t x14 = 1.0 / (x11 * x9 - x12 * x7 - x13 * x9 + x2 * x6 + x3 * x7 * x8 - x5 * x6);
    const real_t x15 = x14 * (x2 - x5);
    const real_t x16 = x14 * (-x12 + x3 * x8);
    const real_t x17 = x14 * (x11 - x13);
    const real_t x18 = -x15 - x16 - x17;
    const real_t x19 = x14 * (-x1 * x7 + x4 * x9);
    const real_t x20 = 0.5 * ux[1];
    const real_t x21 = x14 * (x1 * x6 - x8 * x9);
    const real_t x22 = 0.5 * ux[2];
    const real_t x23 = x14 * (-x4 * x6 + x7 * x8);
    const real_t x24 = 0.5 * ux[3];
    const real_t x25 = 0.5 * x15;
    const real_t x26 = 0.5 * x16;
    const real_t x27 = 0.5 * x17;
    const real_t x28 = -x19 - x21 - x23;
    const real_t x29 = 0.5 * ux[0];
    const real_t x30 = 0.5 * x18;
    const real_t x31 = x14 * (-x0 * x9 + x3 * x7);
    const real_t x32 = x14 * (x10 * x9 - x3 * x6);
    const real_t x33 = x14 * (x0 * x6 - x10 * x7);
    const real_t x34 = -x31 - x32 - x33;
    shear[0] = 1.0 * ux[0] * x18 + 1.0 * ux[1] * x15 + 1.0 * ux[2] * x16 + 1.0 * ux[3] * x17;
    shear[1] = uy[0] * x30 + uy[1] * x25 + uy[2] * x26 + uy[3] * x27 + x19 * x20 + x21 * x22 + x23 * x24 + x28 * x29;
    shear[2] = uz[0] * x30 + uz[1] * x25 + uz[2] * x26 + uz[3] * x27 + x20 * x31 + x22 * x32 + x24 * x33 + x29 * x34;
    shear[3] = 1.0 * uy[0] * x28 + 1.0 * uy[1] * x19 + 1.0 * uy[2] * x21 + 1.0 * uy[3] * x23;
    shear[4] = 0.5 * uy[0] * x34 + 0.5 * uy[1] * x31 + 0.5 * uy[2] * x32 + 0.5 * uy[3] * x33 + 0.5 * uz[0] * x28 +
               0.5 * uz[1] * x19 + 0.5 * uz[2] * x21 + 0.5 * uz[3] * x23;
    shear[5] = 1.0 * uz[0] * x34 + 1.0 * uz[1] * x31 + 1.0 * uz[2] * x32 + 1.0 * uz[3] * x33;
}

void shear(const ptrdiff_t nelements,
           const ptrdiff_t nnodes,
           idx_t **const SFEM_RESTRICT elems,
           geom_t **const SFEM_RESTRICT xyz,
           const real_t *const SFEM_RESTRICT ux,
           const real_t *const SFEM_RESTRICT uy,
           const real_t *const SFEM_RESTRICT uz,
           real_t *const SFEM_RESTRICT shear_xx,
           real_t *const SFEM_RESTRICT shear_xy,
           real_t *const SFEM_RESTRICT shear_xz,
           real_t *const SFEM_RESTRICT shear_yy,
           real_t *const SFEM_RESTRICT shear_yz,
           real_t *const SFEM_RESTRICT shear_zz) {
    SFEM_UNUSED(nnodes);

    idx_t ev[4];
    real_t element_vector[4];

    real_t element_ux[4];
    real_t element_uy[4];
    real_t element_uz[4];
    real_t element_shear[6];

    for (ptrdiff_t i = 0; i < nelements; ++i) {
#pragma unroll(4)
        for (int v = 0; v < 4; ++v) {
            ev[v] = elems[v][i];
        }

        for (int v = 0; v < 4; ++v) {
            element_ux[v] = ux[ev[v]];
        }

        for (int v = 0; v < 4; ++v) {
            element_uy[v] = uy[ev[v]];
        }

        for (int v = 0; v < 4; ++v) {
            element_uz[v] = uz[ev[v]];
        }

        // Element indices
        const idx_t i0 = ev[0];
        const idx_t i1 = ev[1];
        const idx_t i2 = ev[2];
        const idx_t i3 = ev[3];

        shear_kernel(
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
            element_ux,
            element_uy,
            element_uz,
            // Output
            element_shear);

        shear_xx[i] = element_shear[0];
        shear_xy[i] = element_shear[1];
        shear_xz[i] = element_shear[2];
        shear_yy[i] = element_shear[3];
        shear_yz[i] = element_shear[4];
        shear_zz[i] = element_shear[5];
    }
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

    if (argc < 5) {
        fprintf(stderr, "usage: %s <folder> <ux.raw> <uy.raw> <uz.raw> <shear_prefix>\n", argv[0]);
        return EXIT_FAILURE;
    }

    const char *folder = argv[1];
    const char *path_u[3] = {argv[2], argv[3], argv[4]};
    const char *output_prefix = argv[5];

    printf("%s %s %s %s %s %s\n", argv[0], folder, path_u[0], path_u[1], path_u[2], output_prefix);

    double tick = MPI_Wtime();

    ///////////////////////////////////////////////////////////////////////////////
    // Read data
    ///////////////////////////////////////////////////////////////////////////////

    mesh_t mesh;
    if (mesh_read(comm, folder, &mesh)) {
        return EXIT_FAILURE;
    }

    real_t *u[3];

    ptrdiff_t u_n_local, u_n_global;

    for (int d = 0; d < 3; ++d) {
        array_create_from_file(comm, path_u[d], SFEM_MPI_REAL_T, (void **)&u[d], &u_n_local, &u_n_global);
    }
    
    real_t *shear_6[6];
    for (int d = 0; d < 6; ++d) {
        shear_6[d] = (real_t *)malloc(mesh.nelements * sizeof(real_t));
    }

    shear(mesh.nelements,
          mesh.nnodes,
          mesh.elements,
          mesh.points,
          u[0],
          u[1],
          u[2],
          shear_6[0],
          shear_6[1],
          shear_6[2],
          shear_6[3],
          shear_6[4],
          shear_6[5]);

    char path[2048];
    for (int d = 0; d < 6; ++d) {
        sprintf(path, "%s.%d.raw", output_prefix, d);
        array_write(comm, path, SFEM_MPI_REAL_T, shear_6[d], mesh.nelements, mesh.nelements);
        free(shear_6[d]);
    }

    double tock = MPI_Wtime();

    if (!rank) {
        printf("----------------------------------------\n");
        printf("#elements %ld #nodes %ld\n", (long)mesh.nelements, (long)mesh.nnodes);
        printf("TTS:\t\t\t%g seconds\n", tock - tick);
    }

    return MPI_Finalize();
}
