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
#include "sfem_defs.h"

#include "read_mesh.h"

#include "laplacian.h"
#include "proteus_hex8_laplacian.h"
#include "sfem_hex8_mesh_graph.h"

static SFEM_INLINE void hex8_eval_f(const scalar_t x,
                                    const scalar_t y,
                                    const scalar_t z,
                                    scalar_t *const f) {
    const scalar_t xm = (1 - x);
    const scalar_t ym = (1 - y);
    const scalar_t zm = (1 - z);

    f[0] = xm * ym * zm;  // (0, 0, 0)
    f[1] = x * ym * zm;   // (1, 0, 0)
    f[2] = x * y * zm;    // (1, 1, 0)
    f[3] = xm * y * zm;   // (0, 1, 0)
    f[4] = xm * ym * z;   // (0, 0, 1)
    f[5] = x * ym * z;    // (1, 0, 1)
    f[6] = x * y * z;     // (1, 1, 1)
    f[7] = xm * y * z;    // (0, 1, 1)
}

static SFEM_INLINE int proteus_hex8_lidx(const int L, const int x, const int y, const int z) {
    int Lp1 = L + 1;
    int ret = z * (Lp1 * Lp1) + y * Lp1 + x;
    assert(ret < proteus_hex8_nxe(L));
    assert(ret >= 0);
    return ret;
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

    if (argc < 4) {
        fprintf(stderr, "usage: %s <folder> <x.raw> <y.raw>\n", argv[0]);
        return EXIT_FAILURE;
    }

    int SFEM_REPEAT = 1;
    int SFEM_USE_IDX = 0;

    SFEM_READ_ENV(SFEM_REPEAT, atoi);
    SFEM_READ_ENV(SFEM_USE_IDX, atoi);

    const char *folder = argv[1];
    const char *path_f = argv[2];
    const char *path_output = argv[3];

    printf("%s %s %s %s\n", argv[0], folder, path_f, path_output);

    double tick = MPI_Wtime();

    ///////////////////////////////////////////////////////////////////////////////
    // Set-up (read and init)
    ///////////////////////////////////////////////////////////////////////////////

    mesh_t mesh;
    if (mesh_read(comm, folder, &mesh)) {
        return EXIT_FAILURE;
    }

    int SFEM_ELEMENT_REFINE_LEVEL = 2;
    SFEM_READ_ENV(SFEM_ELEMENT_REFINE_LEVEL, atoi);

    int level = SFEM_ELEMENT_REFINE_LEVEL;

    int SFEM_HEX8_ASSUME_AFFINE = 0;
    SFEM_READ_ENV(SFEM_HEX8_ASSUME_AFFINE, atoi);

    const int nxe = proteus_hex8_nxe(level);
    const int txe = proteus_hex8_txe(level);

    printf("nelements %ld\n", mesh.nelements);
    printf("nnodes    %ld\n", mesh.nnodes);
    printf("nxe       %d\n", nxe);
    printf("txe       %d\n", txe);

    idx_t **elements = 0;

    elements = malloc(nxe * sizeof(idx_t *));
    for (int d = 0; d < nxe; d++) {
        elements[d] = malloc(mesh.nelements * sizeof(idx_t));
    }

    for (int d = 0; d < nxe; d++) {
        for (ptrdiff_t i = 0; i < mesh.nelements; i++) {
            elements[d][i] = -1;
        }
    }

    ptrdiff_t n_unique_nodes, interior_start;
    proteus_hex8_create_full_idx(level, &mesh, elements, &n_unique_nodes, &interior_start);

    // ///////////////////////////////////////////////////////////////////////////////
    // Generate explicit hex8 micro-mesh
    // ///////////////////////////////////////////////////////////////////////////////

    idx_t **hex8_elements = malloc(8 * sizeof(idx_t *));
    for (int d = 0; d < 8; d++) {
        hex8_elements[d] = malloc(mesh.nelements * txe * sizeof(idx_t));
    }

    // Elements

    int lnode[8];
    for (int zi = 0; zi < level; zi++) {
        for (int yi = 0; yi < level; yi++) {
            for (int xi = 0; xi < level; xi++) {
                lnode[0] = proteus_hex8_lidx(level, xi, yi, zi);
                lnode[1] = proteus_hex8_lidx(level, xi + 1, yi, zi);
                lnode[2] = proteus_hex8_lidx(level, xi + 1, yi + 1, zi);
                lnode[3] = proteus_hex8_lidx(level, xi, yi + 1, zi);

                lnode[4] = proteus_hex8_lidx(level, xi, yi, zi + 1);
                lnode[5] = proteus_hex8_lidx(level, xi + 1, yi, zi + 1);
                lnode[6] = proteus_hex8_lidx(level, xi + 1, yi + 1, zi + 1);
                lnode[7] = proteus_hex8_lidx(level, xi, yi + 1, zi + 1);

                int le = zi * level * level + yi * level + xi;

                for (int l = 0; l < 8; l++) {
                    for (ptrdiff_t e = 0; e < mesh.nelements; e++) {
                        hex8_elements[l][e * nxe + le] = elements[lnode[l]][e];
                    }
                }
            }
        }
    }

    geom_t **hex8_points = malloc(3 * sizeof(geom_t *));
    for (int d = 0; d < 3; d++) {
        hex8_points[d] = calloc(n_unique_nodes, sizeof(geom_t));
    }

    const int proteus_to_std_hex8_corners[8] = {// Bottom
                                                proteus_hex8_lidx(level, 0, 0, 0),
                                                proteus_hex8_lidx(level, level, 0, 0),
                                                proteus_hex8_lidx(level, level, level, 0),
                                                proteus_hex8_lidx(level, 0, level, 0),

                                                // Top
                                                proteus_hex8_lidx(level, 0, 0, level),
                                                proteus_hex8_lidx(level, level, 0, level),
                                                proteus_hex8_lidx(level, level, level, level),
                                                proteus_hex8_lidx(level, 0, level, level)};

    // Nodes
    const scalar_t h = 1. / level;
    scalar_t f[8];
    for (int zi = 0; zi < level + 1; zi++) {
        for (int yi = 0; yi < level + 1; yi++) {
            for (int xi = 0; xi < level + 1; xi++) {
                const scalar_t x = xi * h;
                const scalar_t y = yi * h;
                const scalar_t z = zi * h;

                hex8_eval_f(x, y, z, f);

                int lidx = proteus_hex8_lidx(level, xi, yi, zi);

                for (int d = 0; d < 3; d++) {
                    for (ptrdiff_t e = 0; e < mesh.nelements; e++) {
                        scalar_t acc = 0;

                        for (int lnode = 0; lnode < 8; lnode++) {
                            const int corner_idx = proteus_to_std_hex8_corners[lnode];
                            geom_t p = mesh.points[d][elements[corner_idx][e]];
                            acc += p * f[lnode];
                        }

                        idx_t node = elements[lidx][e];
                        hex8_points[d][node] = acc;
                    }
                }
            }
        }
    }

    // ///////////////////////////////////////////////////////////////////////////////
    // Free resources
    // ///////////////////////////////////////////////////////////////////////////////

    mesh_destroy(&mesh);

    for (int d = 0; d < nxe; d++) {
        free(elements[d]);
    }

    free(elements);

    //

    for (int d = 0; d < 8; d++) {
        free(hex8_elements[d]);
    }

    free(hex8_elements);

    for (int d = 0; d < 3; d++) {
        free(hex8_points[d]);
    }

    free(hex8_points);

    // ///////////////////////////////////////////////////////////////////////////////
    // Stats
    // ///////////////////////////////////////////////////////////////////////////////

    double tock = MPI_Wtime();
    float TTS = tock - tick;
    return MPI_Finalize();
}
