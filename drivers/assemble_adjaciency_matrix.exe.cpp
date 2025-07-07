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

#include "laplacian.h"
#include "mass.h"

#include "dirichlet.h"
#include "neumann.h"

#include "read_mesh.h"

#include "sfem_API.hpp"

ptrdiff_t read_file(MPI_Comm comm, const char *path, void **data) {
    MPI_Status status;
    MPI_Offset nbytes;
    MPI_File file;
    MPI_CATCH_ERROR(MPI_File_open(comm, path, MPI_MODE_RDONLY, MPI_INFO_NULL, &file));
    MPI_CATCH_ERROR(MPI_File_get_size(file, &nbytes));
    *data = malloc(nbytes);

    MPI_CATCH_ERROR(MPI_File_read_at_all(file, 0, *data, nbytes, MPI_CHAR, &status));
    return nbytes;
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

    if (argc < 2) {
        fprintf(stderr, "usage: %s <folder> [output_folder=./]\n", argv[0]);
        return EXIT_FAILURE;
    }

    const char *output_folder = "./";
    if (argc > 2) {
        output_folder = argv[2];
    }

    printf("%s %s %s\n", argv[0], argv[1], output_folder);

    int SFEM_EXPORT_FP32 = 0;
    int SFEM_REMOVE_DIAGONAL = 1;
    int SFEM_GRAPH_LAPLACIAN = 0;
    int SFEM_NORMALIZE_ROWS = 0;
    int SFEM_REMOVE_LOWER_TRIANGULAR = 0;
    int SFEM_NEGATE_LOWER_TRIANGULAR = 0;

    // Ideas
    // 1) Use the concept of distance and modulo
    // 2) Construct a breadth-first indexing
    int SFEM_DIRECTED = 0;

    // These to options do not work
    SFEM_READ_ENV(SFEM_DIRECTED, atoi);
    SFEM_READ_ENV(SFEM_REMOVE_LOWER_TRIANGULAR, atoi);
    SFEM_READ_ENV(SFEM_NEGATE_LOWER_TRIANGULAR, atoi);

    SFEM_READ_ENV(SFEM_EXPORT_FP32, atoi);
    SFEM_READ_ENV(SFEM_REMOVE_DIAGONAL, atoi);
    SFEM_READ_ENV(SFEM_GRAPH_LAPLACIAN, atoi);
    SFEM_READ_ENV(SFEM_NORMALIZE_ROWS, atoi);

    MPI_Datatype value_type = SFEM_EXPORT_FP32 ? MPI_FLOAT : MPI_DOUBLE;

    double tick = MPI_Wtime();

    ///////////////////////////////////////////////////////////////////////////////
    // Read data
    ///////////////////////////////////////////////////////////////////////////////

    const char *folder = argv[1];

    auto mesh = sfem::Mesh::create_from_file(sfem::Communicator::wrap(comm), folder);
    const ptrdiff_t n_nodes = mesh->n_nodes();
    const ptrdiff_t n_elements = mesh->n_elements();

    double tack = MPI_Wtime();
    printf("assemble.c: read\t\t%g seconds\n", tack - tick);

    ///////////////////////////////////////////////////////////////////////////////
    // Build CRS graph
    ///////////////////////////////////////////////////////////////////////////////

    ptrdiff_t nnz = 0;
    count_t *rowptr = 0;
    idx_t *colidx = 0;
    real_t *values = 0;

    build_crs_graph_for_elem_type(
        mesh->element_type(), n_elements, n_nodes, mesh->elements()->data(), &rowptr, &colidx);

    nnz = rowptr[n_nodes];
    values = (real_t *)malloc(nnz * sizeof(real_t));

    if (SFEM_DIRECTED) {
        printf("SFEM_DIRECTED\n");
        for (ptrdiff_t i = 0; i < n_nodes; i++) {
            for (count_t k = rowptr[i]; k < rowptr[i + 1]; k++) {
                idx_t col = colidx[k];
                if (col != i) {
                    idx_t ii = (i + n_nodes / 2) % n_nodes;
                    idx_t jj = (col + n_nodes / 2) % n_nodes;

                    if (col < i) {
                        values[k] = 1;
                    }
                    // else {
                    //     values[k] = -1;
                    // }
                }
            }
        }
    } else {
        if (SFEM_GRAPH_LAPLACIAN) {
            for (count_t i = 0; i < nnz; i++) {
                values[i] = 1;
            }

            for (ptrdiff_t i = 0; i < n_nodes; i++) {
                real_t row_sum = 0;
                for (count_t k = rowptr[i]; k < rowptr[i + 1]; k++) {
                    row_sum += values[k];
                }

                for (count_t k = rowptr[i]; k < rowptr[i + 1]; k++) {
                    idx_t col = colidx[k];
                    if (col == i) {
                        values[k] = 1 - row_sum;
                    }
                }
            }

        } else {
            for (count_t i = 0; i < nnz; i++) {
                values[i] = 1;
            }

            if (SFEM_REMOVE_DIAGONAL) {
                for (ptrdiff_t i = 0; i < n_nodes; i++) {
                    for (count_t k = rowptr[i]; k < rowptr[i + 1]; k++) {
                        idx_t col = colidx[k];

                        if (col == i) {
                            values[k] = 0;
                        }
                    }
                }
            }

            if (SFEM_REMOVE_LOWER_TRIANGULAR) {
                for (ptrdiff_t i = 0; i < n_nodes; i++) {
                    for (count_t k = rowptr[i]; k < rowptr[i + 1]; k++) {
                        idx_t col = colidx[k];
                        if (col < i) {
                            values[k] = 0;
                        }
                    }
                }
            }

            if (SFEM_NEGATE_LOWER_TRIANGULAR) {
                for (ptrdiff_t i = 0; i < n_nodes; i++) {
                    for (count_t k = rowptr[i]; k < rowptr[i + 1]; k++) {
                        idx_t col = colidx[k];
                        if (col < i) {
                            values[k] = -values[k];
                        }
                    }
                }
            }

            if (SFEM_NORMALIZE_ROWS) {
                for (ptrdiff_t i = 0; i < n_nodes; i++) {
                    real_t row_sum = 0;
                    for (count_t k = rowptr[i]; k < rowptr[i + 1]; k++) {
                        row_sum += values[k];
                    }

                    for (count_t k = rowptr[i]; k < rowptr[i + 1]; k++) {
                        values[k] /= row_sum;
                    }
                }
            }
        }
    }

    double tock = MPI_Wtime();
    printf("assemble.c: build crs\t\t%g seconds\n", tock - tack);
    tack = tock;

    ///////////////////////////////////////////////////////////////////////////////
    // Write CRS matrix
    ///////////////////////////////////////////////////////////////////////////////

    {
        crs_t crs_out;
        crs_out.rowptr = (char *)rowptr;
        crs_out.colidx = (char *)colidx;
        crs_out.values = (char *)values;
        crs_out.grows = n_nodes;
        crs_out.lrows = n_nodes;
        crs_out.lnnz = nnz;
        crs_out.gnnz = nnz;
        crs_out.start = 0;
        crs_out.rowoffset = 0;
        crs_out.rowptr_type = SFEM_MPI_COUNT_T;
        crs_out.colidx_type = SFEM_MPI_IDX_T;
        crs_out.values_type = value_type;
        crs_write_folder(comm, output_folder, &crs_out);
    }

    tock = MPI_Wtime();
    printf("assemble.c: write\t\t%g seconds\n", tock - tack);
    tack = tock;

    ///////////////////////////////////////////////////////////////////////////////
    // Free resources
    ///////////////////////////////////////////////////////////////////////////////

    free(rowptr);
    free(colidx);
    free(values);

    tock = MPI_Wtime();

    if (!rank) {
        printf("----------------------------------------\n");
        printf("#elements %ld #nodes %ld #nz %ld\n", (long)n_elements, (long)n_nodes, (long)nnz);
        printf("TTS:\t\t\t%g seconds\n", tock - tick);
    }

    return MPI_Finalize();
}
