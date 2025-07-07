#include "sfem_API.hpp"

#include "matrixio_array.h"
#include "sfem_defs.h"
#include "sfem_mesh_write.h"
#include "tet4_inline_cpu.h"

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

    if (argc != 3) {
        fprintf(stderr, "usage: %s <folder> <output>\n", argv[0]);
        return EXIT_FAILURE;
    }

    const char *output_folder = argv[2];
    const char *folder = argv[1];
    auto m = sfem::Mesh::create_from_file(sfem::Communicator::wrap(comm), folder);

    int nxe = elem_num_nodes((enum ElemType)m->element_type());
    const auto elements = m->elements()->data();
    const auto points = m->points()->data();
    const ptrdiff_t nelements = m->n_elements();

    ptrdiff_t n_reorders = 0;
    for (ptrdiff_t e = 0; e < nelements; e++) {
        idx_t ev[8];
        for (int v = 0; v < 8; v++) {
            ev[v] = elements[v][e];
        }

        geom_t x[8], y[8], z[8];
        for (int v = 0; v < 8; v++) {
            x[v] = points[0][ev[v]];
            y[v] = points[1][ev[v]];
            z[v] = points[2][ev[v]];
        }

        jacobian_t adjugate[9];
        jacobian_t determinant;

        tet4_adjugate_and_det(x[0],
                              x[1],
                              x[3],
                              x[4],
                              y[0],
                              y[1],
                              y[3],
                              y[4],
                              z[0],
                              z[1],
                              z[3],
                              z[4],
                              adjugate,
                              &determinant);

        jacobian_t inverse[9];

        for (int d = 0; d < 9; d++) {
            inverse[d] = adjugate[d] / determinant;
        }

        geom_t rx[8], ry[8], rz[8];
        for (int v = 0; v < 8; v++) {
            const geom_t xv = x[v] - x[0];
            const geom_t yv = y[v] - y[0];
            const geom_t zv = z[v] - z[0];
            rx[v] = (inverse[0] * xv + inverse[1] * yv + inverse[2] * zv);
            ry[v] = (inverse[3] * xv + inverse[4] * yv + inverse[5] * zv);
            rz[v] = (inverse[6] * xv + inverse[7] * yv + inverse[8] * zv);
        }

        int order[8];
        geom_t dist[8];
        for (int v = 0; v < 8; v++) {
            const geom_t xv = rx[v];
            const geom_t yv = ry[v];
            const geom_t zv = rz[v];

            dist[0] = xv * xv + yv * yv + zv * zv;                                // 0) (0, 0, 0)
            dist[1] = (1 - xv) * (1 - xv) + (-yv) * (-yv) + (-zv) * (-zv);        // 1) (1, 0, 0)
            dist[2] = (1 - xv) * (1 - xv) + (1 - yv) * (1 - yv) + (-zv) * (-zv);  // 2) (1, 1, 0)
            dist[3] = (-xv) * (-xv) + (1 - yv) * (1 - yv) + (-zv) * (-zv);        // 3) (0, 1, 0)
            dist[4] = (-xv) * (-xv) + (-yv) * (-yv) + (1 - zv) * (1 - zv);        // 4) (0, 0, 1)
            dist[5] = (1 - xv) * (1 - xv) + (-yv) * (-yv) + (1 - zv) * (1 - zv);  // 5) (1, 0, 1)
            dist[6] = (1 - xv) * (1 - xv) + (1 - yv) * (1 - yv) +
                      (1 - zv) * (1 - zv);                                        // 6) (1, 1, 1)
            dist[7] = (-xv) * (-xv) + (1 - yv) * (1 - yv) + (1 - zv) * (1 - zv);  // 7) (0, 1, 1)

            int arg_min = 0;
            geom_t dist_min = dist[0];

            for (int l = 1; l < 8; l++) {
                if (dist_min > dist[l]) {
                    arg_min = l;
                    dist_min = dist[l];
                }
            }

            order[v] = arg_min;
        }

        bool ok = true;
        for (int v = 0; v < 8; v++) {
            for (int l = v + 1; l < 8; l++) {
                if (order[v] == order[l]) {
                    fprintf(stderr, "Badly scaled element cannot be fixed! e=%ld\n", e);
                    assert(false);
                    ok = false;
                }
            }
        }

        bool has_reordered = false;
        for(int v = 0; v < 8; v++) {
            has_reordered |= order[v] != v;
        }

        n_reorders += has_reordered;

        if (ok || !has_reordered) {
            idx_t reordering[8];

            for (int v = 0; v < 8; v++) {
                reordering[order[v]] = ev[v];
            }

            for (int v = 0; v < 8; v++) {
                elements[v][e] = reordering[v];
            }
        }
    }   

    printf("n_reorders= %ld\n", n_reorders);

    if(n_reorders)
        m->write(output_folder);
    return MPI_Finalize();
}
