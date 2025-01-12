#ifdef _WIN32
#define _USE_MATH_DEFINES
#endif

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
// #include <unistd.h>

#include "array_dtof.h"
#include "matrixio_array.h"
#include "matrixio_crs.h"
#include "utils.h"

#include "crs_graph.h"
#include "sfem_base.h"
#include "sfem_defs.h"
#include "sfem_vec.h"
#include "sortreduce.h"

#include "mass.h"

#include "dirichlet.h"
#include "neumann.h"

#include "read_mesh.h"

static SFEM_INLINE real_t ux1(const real_t x, const real_t y) {
    return x * x * (1 - x) * (1 - x) * 2 * y * (1 - y) * (2 * y - 1);
}

static SFEM_INLINE real_t uy1(const real_t x, const real_t y) {
    return y * y * (1 - y) * (1 - y) * 2 * x * (1 - x) * (1 - 2 * x);
}

static SFEM_INLINE real_t p1(const real_t x, const real_t y) {
    return x * (1 - x) * (1 - y) - 1. / 12;
}

static SFEM_INLINE real_t ux2(const real_t x, const real_t y) {
    const real_t x2 = x * x;
    const real_t x3 = x2 * x;
    const real_t x4 = x3 * x;

    const real_t y2 = y * y;
    const real_t y3 = y2 * y;

    return (x2 - 2 * x3 + x4) * (2 * y - 6 * y2 + 4 * y3);
}

static SFEM_INLINE real_t uy2(const real_t x, const real_t y) {
    const real_t x2 = x * x;
    const real_t x3 = x2 * x;

    const real_t y2 = y * y;
    const real_t y3 = y2 * y;
    const real_t y4 = y3 * y;

    return -(2 * x - 6 * x2 + 4 * x3) * (y2 - 2 * y3 + y4);
}

static SFEM_INLINE real_t p2(const real_t x, const real_t y) { return (x + y - 1) / 24; }

static SFEM_INLINE real_t ux3(const real_t x, const real_t y) {
    return sin(2 * M_PI * y) * (1 - cos(2 * M_PI * x));
}

static SFEM_INLINE real_t uy3(const real_t x, const real_t y) {
    return sin(2 * M_PI * x) * (cos(2 * M_PI * y) - 1);
}

static SFEM_INLINE real_t p3(const real_t x, const real_t y) {
    return 2 * M_PI * (cos(2 * M_PI * y) - cos(2 * M_PI * x));
}

void stokes_ref_sol(const int tp_num,
                    const real_t mu,
                    const ptrdiff_t nnodes,
                    geom_t **const points,
                    real_t *const ux,
                    real_t *const uy,
                    real_t *const p) {
    for (ptrdiff_t i = 0; i < nnodes; ++i) {
        const geom_t x = points[0][i];
        const geom_t y = points[1][i];

        switch (tp_num) {
            case 1: {
                ux[i] = ux1(x, y);
                uy[i] = uy1(x, y);
                p[i] = p1(x, y);
                break;
            }
            case 2: {
                ux[i] = ux2(x, y);
                uy[i] = uy2(x, y);
                p[i] = p2(x, y);
                break;
            }
            case 3: {
                ux[i] = ux3(x, y);
                uy[i] = uy3(x, y);
                p[i] = p3(x, y);
                break;
            }
            default: {
                assert(0);
                break;
            }
        }
    }
}

// void tri3_stokes_check(const int tp_num,
//                        const real_t mu,
//                        const ptrdiff_t nelements,
//                        const ptrdiff_t nnodes,
//                        idx_t **const elems,
//                        geom_t **const points,
//                        const real_t *const ux,
//                        const real_t *const uy,
//                        const real_t *const p) {
//     SFEM_UNUSED(nnodes);

//     double tick = MPI_Wtime();

//     static const int n_vars = 3;
//     static const int ndofs = 3;
//     static const int rows = 9;
//     static const int cols = 9;

//     idx_t ev[3];
//     idx_t ks[3];
//     real_t element_vector[3 * 3];
//     real_t xx[3];
//     real_t yy[3];

//     real_t err_ux = 0;
//     real_t err_uy = 0;
//     real_t err_p = 0;

//     for (ptrdiff_t i = 0; i < nelements; ++i) {
// #pragma unroll(3)
//         for (int v = 0; v < 3; ++v) {
//             ev[v] = elems[v][i];
//         }

//         // Element indices
//         const idx_t i0 = ev[0];
//         const idx_t i1 = ev[1];
//         const idx_t i2 = ev[2];

//         const real_t x0 = points[0][i0];
//         const real_t x1 = points[0][i1];
//         const real_t x2 = points[0][i2];

//         const real_t y0 = points[1][i0];
//         const real_t y1 = points[1][i1];
//         const real_t y2 = points[1][i2];

//         xx[0] = x0;
//         yy[0] = y0;
//         xx[1] = x1;
//         yy[1] = y1;
//         xx[2] = x2;
//         yy[2] = y2;

//         real_t dux = 0;
//         real_t duy = 0;
//         real_t dp = 0;

//         for (int ii = 0; ii < 3; ii++) {
//             real_t aux = 0;
//             real_t auy = 0;
//             real_t ap = 0;

//             switch (tp_num) {
//                 case 1: {
//                     aux = ux1(xx[ii], yy[ii]);
//                     auy = uy1(xx[ii], yy[ii]);
//                     ap = p1(xx[ii], yy[ii]);
//                     break;
//                 }
//                 case 2: {
//                     aux = ux2(xx[ii], yy[ii]);
//                     auy = uy2(xx[ii], yy[ii]);
//                     ap = p2(xx[ii], yy[ii]);
//                     break;
//                 }
//                 default: {
//                     assert(0);
//                     break;
//                 }
//             }

//             dux += sqrt((aux - ux[ev[ii]]) * (aux - ux[ev[ii]]));  // / (aux * aux);
//             duy += sqrt((auy - uy[ev[ii]]) * (auy - uy[ev[ii]]));  // / (auy * auy);
//             dp += sqrt((ap - p[ev[ii]]) * (ap - p[ev[ii]]));       // / (ap * ap));
//         }

//         dux /= 3;
//         duy /= 3;
//         dp /= 3;

//         err_ux += dux;
//         err_uy += duy;
//         err_p += dp;
//     }

//     err_ux /= nelements;
//     err_uy /= nelements;
//     err_p /= nelements;

//     printf("err = (%g, %g, %g)\n", err_ux, err_uy, err_p);
// }

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

    if (argc != 5) {
        // fprintf(stderr, "usage: %s <mesh> <ux.raw> <uy.raw> <uz.raw> <p.raw>\n", argv[0]);
        fprintf(stderr, "usage: %s <mesh> <ux.raw> <uy.raw> <p.raw>\n", argv[0]);
        return EXIT_FAILURE;
    }

    const char *path_ux = argv[2];
    const char *path_uy = argv[3];
    // const char *path_uz = argv[4];
    // const char *path_p = argv[5];

    const char *path_p = argv[4];

    double tick = MPI_Wtime();

    ///////////////////////////////////////////////////////////////////////////////
    // Read data
    ///////////////////////////////////////////////////////////////////////////////

    const char *folder = argv[1];

    mesh_t mesh;
    if (mesh_read(comm, folder, &mesh)) {
        return EXIT_FAILURE;
    }

    if (mesh.element_type != TRI3) {
        fprintf(stderr, "element_type must be TRI3\n");
        return EXIT_FAILURE;
    }

    real_t *ux = (real_t *)calloc(mesh.nnodes, sizeof(real_t));
    real_t *uy = (real_t *)calloc(mesh.nnodes, sizeof(real_t));
    real_t *p = (real_t *)calloc(mesh.nnodes, sizeof(real_t));

    // Optional params
    real_t SFEM_MU = 1;
    int SFEM_PROBLEM_TYPE = 1;

    SFEM_READ_ENV(SFEM_PROBLEM_TYPE, atoi);
    SFEM_READ_ENV(SFEM_MU, atof);

    if (rank == 0) {
        printf(
            "----------------------------------------\n"
            "Options:\n"
            "----------------------------------------\n"
            "- SFEM_PROBLEM_TYPE=%d\n"
            "- SFEM_MU=%g\n"
            "----------------------------------------\n",
            SFEM_PROBLEM_TYPE,
            SFEM_MU);
    }

    stokes_ref_sol(SFEM_PROBLEM_TYPE, SFEM_MU, mesh.nnodes, mesh.points, ux, uy, p);

    array_write(comm, path_ux, SFEM_MPI_REAL_T, ux, mesh.nnodes, mesh.nnodes);
    array_write(comm, path_uy, SFEM_MPI_REAL_T, uy, mesh.nnodes, mesh.nnodes);
    array_write(comm, path_p, SFEM_MPI_REAL_T, p, mesh.nnodes, mesh.nnodes);

    ///////////////////////////////////////////////////////////////////////////////
    // Free resources
    ///////////////////////////////////////////////////////////////////////////////

    ptrdiff_t nelements = mesh.nelements;
    ptrdiff_t nnodes = mesh.nnodes;

    mesh_destroy(&mesh);

    free(ux);
    free(uy);
    free(p);

    double tock = MPI_Wtime();
    if (!rank) {
        printf("----------------------------------------\n");
        printf("#elements %ld #nodes %ld\n", (long)nelements, (long)nnodes);
        printf("TTS:\t\t\t%g seconds\n", tock - tick);
    }

    return MPI_Finalize();
}
