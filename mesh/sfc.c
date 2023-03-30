#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../matrix.io/array_dtof.h"
#include "../matrix.io/matrixio_array.h"
#include "../matrix.io/matrixio_crs.h"
#include "../matrix.io/utils.h"

#include "crs_graph.h"
#include "read_mesh.h"
#include "sfem_base.h"
#include "sfem_mesh_write.h"

#include "argsort.h"

#ifdef DSFEM_ENABLE_MPI_SORT
#include "mpi-sort.h"
#endif

typedef uint32_t sfc_t;
#define SFEM_MPI_SFC_T MPI_UNSIGNED
#define sort_function argsort_u32

// https://mathworld.wolfram.com/FiedlerVector.html

// typedef uint64_t sfc_t;
// #define SFEM_MPI_SFC_T MPI_UNSIGNED_LONG
// #define sort_function argsort_u64

// Space-filling curve mapping function for 3D points
static SFEM_INLINE sfc_t morton3d(sfc_t x, sfc_t y, sfc_t z) {
    x &= 0x3FFFFFFF;
    y &= 0x3FFFFFFF;
    z &= 0x3FFFFFFF;
    x = (x | (x << 16)) & 0x030000FF;
    x = (x | (x << 8)) & 0x0300F00F;
    x = (x | (x << 4)) & 0x030C30C3;
    x = (x | (x << 2)) & 0x09249249;
    y = (y | (y << 16)) & 0x030000FF;
    y = (y | (y << 8)) & 0x0300F00F;
    y = (y | (y << 4)) & 0x030C30C3;
    y = (y | (y << 2)) & 0x09249249;
    z = (z | (z << 16)) & 0x030000FF;
    z = (z | (z << 8)) & 0x0300F00F;
    z = (z | (z << 4)) & 0x030C30C3;
    z = (z | (z << 2)) & 0x09249249;
    return x | (y << 1) | (z << 2);
}

static SFEM_INLINE sfc_t hilbert3d(sfc_t x, sfc_t y, sfc_t z) {
    sfc_t hilbert = 0;
    for (int i = 31; i >= 0; i--) {
        sfc_t mask = 1 << i;
        sfc_t rx = (x & mask) ? 1 : 0;
        sfc_t ry = (y & mask) ? 1 : 0;
        sfc_t rz = (z & mask) ? 1 : 0;
        hilbert = (hilbert << 3) | (rx ^ ry ^ rz);
        if (rz == 1) {
            hilbert = (hilbert << 2) | (rx ^ ry ^ 1);
            hilbert = (hilbert << 2) | (rx ^ 1);
            hilbert ^= 0xAAAAAAAB;
        } else {
            hilbert = (hilbert << 2) | (ry ^ rz);
            hilbert = (hilbert << 2) | rx;
            hilbert ^= 0x55555555;
        }
    }
    return hilbert;
}

// sfc_t morton3d(uint32_t x, uint32_t y, uint32_t z) {
//     sfc_t morton = 0;
//     for (int i = 0; i < 21; i++) {
//         morton |= ((sfc_t)(x & 1) << (3*i + 2))
//                 | ((sfc_t)(y & 1) << (3*i + 1))
//                 | ((sfc_t)(z & 1) << (3*i));
//         x >>= 1;
//         y >>= 1;
//         z >>= 1;
//     }
//     return morton;
// }

// sfc_t hilbert3d(uint32_t x, uint32_t y, uint32_t z) {
//     sfc_t hilbert = 0;
//     for (int i = 31; i >= 0; i--) {
//         sfc_t mask = 1ull << i;
//         sfc_t rx = (x & mask) ? 1ull : 0ull;
//         sfc_t ry = (y & mask) ? 1ull : 0ull;
//         sfc_t rz = (z & mask) ? 1ull : 0ull;
//         hilbert = (hilbert << 3) | (rx ^ ry ^ rz);
//         if (rz == 1ull) {
//             hilbert = (hilbert << 2) | (rx ^ ry ^ 1ull);
//             hilbert = (hilbert << 2) | (rx ^ 1ull);
//             hilbert ^= 0xAAAAAAAABBBBBBBBull;
//         } else {
//             hilbert = (hilbert << 2) | (ry ^ rz);
//             hilbert = (hilbert << 2) | rx;
//             hilbert ^= 0x5555555555555555ull;
//         }
//     }
//     return hilbert;
// }

// #define fun_sfc hilbert3d
// #define sfc_urange 32u

#define fun_sfc morton3d
#define sfc_urange 1024u

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    MPI_Comm comm = MPI_COMM_WORLD;

    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    if (argc != 3) {
        if (!rank) {
            fprintf(stderr, "usage: %s <folder> <output_folder>\n", argv[0]);
        }

        return EXIT_FAILURE;
    }

#ifndef DSFEM_ENABLE_MPI_SORT
    if (size > 1) {
        if (!rank) {
            fprintf(stderr, "Parallel runs not supported. Compile with mpi-sort\n");
        }

        MPI_Abort(comm, -1);
    }
#endif

    const char *folder = argv[1];
    const char *output_folder = argv[2];

    if (!rank) {
        printf("%s %s %s\n", argv[0], folder, output_folder);
    }

    double tick = MPI_Wtime();

    mesh_t mesh;
    if (mesh_read(comm, folder, &mesh)) {
        return EXIT_FAILURE;
    }

    sfc_t *sfc = (sfc_t *)malloc(mesh.n_owned_elements * sizeof(sfc_t));
    memset(sfc, 0, mesh.n_owned_elements * sizeof(sfc_t));

    idx_t *idx = (idx_t *)malloc(mesh.n_owned_elements * sizeof(idx_t));

    geom_t box_min[3], box_max[3];
    for (int coord = 0; coord < mesh.spatial_dim; coord++) {
        box_min[coord] = mesh.points[coord][0];
        box_max[coord] = mesh.points[coord][0];

        for (ptrdiff_t i = 0; i < mesh.n_owned_nodes; i++) {
            const geom_t x = mesh.points[coord][i];
            box_min[coord] = MIN(box_min[coord], x);
            box_max[coord] = MAX(box_max[coord], x);
        }
    }

    int normalize_with_max_range = 1;

    if (normalize_with_max_range) {
        geom_t m0 = box_min[0], m1 = box_max[1];

        for (int coord = 1; coord < mesh.spatial_dim; coord++) {
            m0 = MIN(m0, box_min[coord]);
            m1 = MAX(m1, box_max[coord]);
        }

        for (int coord = 0; coord < mesh.spatial_dim; coord++) {
            box_min[coord] = m0;
            box_max[coord] = m1;
        }
    }

    sfc_t urange[3] = {sfc_urange, sfc_urange, sfc_urange};

    for (ptrdiff_t i = 0; i < mesh.n_owned_elements; i++) {
        geom_t b[3] = {0, 0, 0};
        const idx_t i0 = mesh.elements[0][i];

        for (int coord = 0; coord < mesh.spatial_dim; coord++) {
            geom_t x = mesh.points[coord][i0];
            x -= box_min[coord];
            x /= box_max[coord] - box_min[coord];
            b[coord] = x;
        }

        for (int d = 1; d < mesh.element_type; d++) {
            const idx_t ii = mesh.elements[d][i];

            for (int coord = 0; coord < mesh.spatial_dim; coord++) {
                geom_t x = mesh.points[coord][ii];
                x -= box_min[coord];
                x /= box_max[coord] - box_min[coord];
                b[coord] = MIN(b[coord], x);
            }
        }

        assert(b[0] >= 0);
        assert(b[0] <= 1);

        assert(b[1] >= 0);
        assert(b[1] <= 1);

        assert(b[2] >= 0);
        assert(b[2] <= 1);

        // sfc[i] = (sfc_t)(b[2] * (geom_t)urange[2]);
        sfc[i] = fun_sfc((sfc_t)(b[0] * (double)urange[0]),  //
                         (sfc_t)(b[1] * (double)urange[1]),  //
                         (sfc_t)(b[2] * (double)urange[2])   //
        );

        // printf("%d -> %g %g %g %d\n", (int)i, (double)b[0], (double)b[1], (double)b[2], (int)sfc[i]);
    }

#ifdef DSFEM_ENABLE_MPI_SORT

    if (size > 1) {
        // TODO

        // MPI_Sort_bykey (
        //     void * sendkeys_destructive,
        //     void * sendvals_destructive,
        //     const int sendcount,
        //     MPI_Datatype keytype,
        //     MPI_Datatype valtype,
        //     void * recvkeys,
        //     void * recvvals,
        //     const int recvcount,
        //     comm);
    } else
#endif
    {
        for (ptrdiff_t i = 0; i < mesh.n_owned_elements; i++) {
            idx[i] = i;
        }

        sort_function(mesh.n_owned_elements, sfc, idx);

        ptrdiff_t buff_size = MAX(mesh.n_owned_elements, mesh.n_owned_nodes) * sizeof(idx_t);
        void *buff = malloc(buff_size);

        // 1) rearrange elements
        {
            idx_t *elem_buff = (idx_t *)buff;
            for (int d = 0; d < mesh.element_type; d++) {
                memcpy(elem_buff, mesh.elements[d], mesh.n_owned_elements * sizeof(idx_t));
                for (ptrdiff_t i = 0; i < mesh.n_owned_elements; i++) {
                    mesh.elements[d][i] = elem_buff[idx[i]];
                }
            }

            const char *SFEM_EXPORT_SFC = 0;
            SFEM_READ_ENV(SFEM_EXPORT_SFC, );
            if (SFEM_EXPORT_SFC) {
                memcpy(elem_buff, sfc, mesh.n_owned_elements * sizeof(sfc_t));
                for (ptrdiff_t i = 0; i < mesh.n_owned_elements; i++) {
                    sfc[i] = elem_buff[idx[i]];
                }

                array_write(comm, SFEM_EXPORT_SFC, SFEM_MPI_SFC_T, sfc, mesh.n_owned_elements, mesh.n_owned_elements);
            }
        }

        // 2) rearrange element_mapping (if the case)
        // TODO

        // 3) rearrange nodes
        idx_t *node_buff = (idx_t *)buff;

        {
            memset(node_buff, 0, mesh.n_owned_nodes * sizeof(idx_t));

            idx_t next_node = 1;
            for (ptrdiff_t i = 0; i < mesh.n_owned_elements; i++) {
                for (int d = 0; d < mesh.element_type; d++) {
                    idx_t i0 = mesh.elements[d][i];

                    if (!node_buff[i0]) {
                        node_buff[i0] = next_node++;
                        assert(next_node - 1 <= mesh.n_owned_nodes);
                    }
                }
            }

            assert(next_node - 1 == mesh.n_owned_nodes);

            for (ptrdiff_t i = 0; i < mesh.n_owned_nodes; i++) {
                assert(node_buff[i] > 0);
                node_buff[i] -= 1;
            }
        }

        // Update e2n
        for (int d = 0; d < mesh.element_type; d++) {
            for (ptrdiff_t i = 0; i < mesh.n_owned_elements; i++) {
                idx_t i0 = mesh.elements[d][i];
                mesh.elements[d][i] = node_buff[i0];
            }
        }

        // update coordinates
        geom_t *x_buff = malloc(mesh.n_owned_nodes * sizeof(geom_t));
        for (int d = 0; d < mesh.spatial_dim; d++) {
            memcpy(x_buff, mesh.points[d], mesh.n_owned_nodes * sizeof(geom_t));

            for (ptrdiff_t i = 0; i < mesh.n_owned_nodes; i++) {
                mesh.points[d][node_buff[i]] = x_buff[i];
            }
        }

        // 4) rearrange (or create node_mapping (for relating data))
        // TODO

        free(buff);
        free(x_buff);
    }

    free(sfc);
    free(idx);

    mesh_write(output_folder, &mesh);
    mesh_destroy(&mesh);
    double tock = MPI_Wtime();

    if (!rank) {
        printf("----------------------------------------\n");
        printf("TTS:\t\t\t%g seconds\n", tock - tick);
    }
}
