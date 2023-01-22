#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../matrix.io/matrixio_array.h"
#include "../matrix.io/matrixio_crs.h"
#include "../matrix.io/utils.h"
#include "../matrix.io/array_dtof.h"

#include "crs_graph.h"
#include "sfem_base.h"

#include "laplacian.h"
#include "mass.h"

#include "read_mesh.h"

ptrdiff_t read_file(MPI_Comm comm, const char *path, void **data) {
    MPI_Status status;
    MPI_Offset nbytes;
    MPI_File file;
    CATCH_MPI_ERROR(MPI_File_open(comm, path, MPI_MODE_RDONLY, MPI_INFO_NULL, &file));
    CATCH_MPI_ERROR(MPI_File_get_size(file, &nbytes));
    *data = malloc(nbytes);

    CATCH_MPI_ERROR(MPI_File_read_at_all(file, 0, *data, nbytes, MPI_CHAR, &status));
    return nbytes;
}

SFEM_INLINE real_t det3(const real_t *mat) {
    return mat[0] * mat[4] * mat[8] + mat[1] * mat[5] * mat[6] + mat[2] * mat[3] * mat[7] - mat[0] * mat[5] * mat[7] -
           mat[1] * mat[3] * mat[8] - mat[2] * mat[4] * mat[6];
}

SFEM_INLINE real_t area3(const real_t left[3], const real_t right[3]) {
    real_t a = (left[1] * right[2]) - (right[1] * left[2]);
    real_t b = (left[2] * right[0]) - (right[2] * left[0]);
    real_t c = (left[0] * right[1]) - (right[0] * left[1]);
    return sqrt(a * a + b * b + c * c);
}

SFEM_INLINE void integrate_neumann(real_t value, real_t dA, real_t *element_vector) {
    element_vector[0] = value * dA;
    element_vector[1] = value * dA;
    element_vector[2] = value * dA;
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
        fprintf(stderr, "usage: %s <folder> [output_folder=./]", argv[0]);
        return EXIT_FAILURE;
    }

    const char *output_folder = "./";
    if (argc > 2) {
        output_folder = argv[2];
    }

    printf("%s %s %s\n", argv[0], argv[1], output_folder);

    int SFEM_LAPLACIAN = 1;
    int SFEM_MASS = 0;
    int SFEM_HANDLE_DIRICHLET = 1;
    int SFEM_EXPORT_FP32 = 0;

    SFEM_READ_ENV(SFEM_LAPLACIAN, atoi);
    SFEM_READ_ENV(SFEM_MASS, atoi);
    SFEM_READ_ENV(SFEM_HANDLE_DIRICHLET, atoi);
    SFEM_READ_ENV(SFEM_EXPORT_FP32, atoi);

    printf("----------------------------------------\n");
    printf("Environment variables:\n- SFEM_LAPLACIAN=%d\n- SFEM_MASS=%d\n- SFEM_HANDLE_DIRICHLET=%d\n- SFEM_EXPORT_FP32=%d\n",
           SFEM_LAPLACIAN,
           SFEM_MASS,
           SFEM_HANDLE_DIRICHLET,
           SFEM_EXPORT_FP32);
    printf("----------------------------------------\n");

    double tick = MPI_Wtime();

    ///////////////////////////////////////////////////////////////////////////////
    // Read data
    ///////////////////////////////////////////////////////////////////////////////

    const char *folder = argv[1];
    char path[1024 * 10];
    ptrdiff_t nnodes = 0;
    geom_t *xyz[3];

    ptrdiff_t nelements = 0;
    idx_t *elems[4];

    if(serial_read_tet_mesh(folder, &nelements, elems, &nnodes, xyz)) {
        return EXIT_FAILURE;
    }

    double tack = MPI_Wtime();
    printf("assemble.c: read\t\t%g seconds\n", tack - tick);

    ///////////////////////////////////////////////////////////////////////////////
    // Build CRS graph
    ///////////////////////////////////////////////////////////////////////////////

    ptrdiff_t nnz = 0;
    idx_t *rowptr = (idx_t *)malloc((nnodes + 1) * sizeof(idx_t));
    idx_t *colidx = 0;
    real_t *values = 0;

    build_crs_graph(nelements, nnodes, elems, &rowptr, &colidx);

    nnz = rowptr[nnodes];
    values = (real_t *)malloc(nnz * sizeof(real_t));
    memset(values, 0, nnz * sizeof(real_t));

    double tock = MPI_Wtime();
    printf("assemble.c: build crs\t\t%g seconds\n", tock - tack);
    tack = tock;

    ///////////////////////////////////////////////////////////////////////////////
    // Operator assembly
    ///////////////////////////////////////////////////////////////////////////////
    if (SFEM_LAPLACIAN) {
        assemble_laplacian(nelements, nnodes, elems, xyz, rowptr, colidx, values);
    }

    if (SFEM_MASS) {
        assemble_mass(nelements, nnodes, elems, xyz, rowptr, colidx, values);
    }

    tock = MPI_Wtime();
    printf("assemble.c: assembly\t\t%g seconds\n", tock - tack);
    tack = tock;

    ///////////////////////////////////////////////////////////////////////////////
    // Boundary conditions
    ///////////////////////////////////////////////////////////////////////////////

    real_t *rhs = (real_t *)malloc(nnodes * sizeof(real_t));
    memset(rhs, 0, nnodes * sizeof(real_t));

    {  // Neumann
        sprintf(path, "%s/on.raw", folder);
        idx_t *faces_neumann = 0;
        ptrdiff_t nfacesx3 = read_file(comm, path, (void **)&faces_neumann);
        idx_t nfaces = (nfacesx3 / 3) / sizeof(idx_t);
        assert(nfaces * 3 * sizeof(idx_t) == nfacesx3);

        real_t u[3], v[3];
        real_t element_vector[3];

        real_t jacobian[3 * 3] = {0, 0, 0, 0, 0, 0, 0, 0, 1};

        real_t value = 1.0;
        // real_t value = 2.0;
        for (idx_t f = 0; f < nfaces; ++f) {
            idx_t i0 = faces_neumann[f * 3];
            idx_t i1 = faces_neumann[f * 3 + 1];
            idx_t i2 = faces_neumann[f * 3 + 2];

            real_t dx = 0;

            if (0) {
                for (int d = 0; d < 3; ++d) {
                    real_t x0 = (real_t)xyz[d][i0];
                    real_t x1 = (real_t)xyz[d][i1];
                    real_t x2 = (real_t)xyz[d][i2];

                    u[d] = x1 - x0;
                    v[d] = x2 - x0;
                }

                dx = area3(u, v) / 2;
            } else {
                // No square roots in this version
                for (int d = 0; d < 3; ++d) {
                    real_t x0 = (real_t)xyz[d][i0];
                    real_t x1 = (real_t)xyz[d][i1];
                    real_t x2 = (real_t)xyz[d][i2];

                    jacobian[d * 3] = x1 - x0;
                    jacobian[d * 3 + 1] = x2 - x0;
                }

                // Orientation of face is not proper
                dx = fabs(det3(jacobian)) / 2;
            }

            assert(dx > 0.);
            integrate_neumann(value, dx, element_vector);

            rhs[i0] += element_vector[0];
            rhs[i1] += element_vector[1];
            rhs[i2] += element_vector[2];
        }

        free(faces_neumann);
    }

    if (SFEM_HANDLE_DIRICHLET) {
        // Dirichlet
        sprintf(path, "%s/zd.raw", folder);
        idx_t *dirichlet_nodes = 0;
        ptrdiff_t nn = read_file(comm, path, (void **)&dirichlet_nodes);
        assert((nn / sizeof(idx_t)) * sizeof(idx_t) == nn);
        nn /= sizeof(idx_t);

        // Set rhs should not be necessary (but let us do it anyway)
        for (idx_t node = 0; node < nn; ++node) {
            idx_t i = dirichlet_nodes[node];
            rhs[i] = 0;
        }

        for (idx_t node = 0; node < nn; ++node) {
            idx_t i = dirichlet_nodes[node];

            idx_t begin = rowptr[i];
            idx_t end = rowptr[i + 1];
            idx_t lenrow = end - begin;
            idx_t *cols = &colidx[begin];
            real_t *row = &values[begin];

            memset(row, 0, sizeof(real_t) * lenrow);

            int k = find_idx(i, cols, lenrow);
            assert(k >= 0);
            row[k] = 1;
        }
    }

    tock = MPI_Wtime();
    printf("assemble.c: boundary\t\t%g seconds\n", tock - tack);
    tack = tock;

    ///////////////////////////////////////////////////////////////////////////////
    // Write CRS matrix and rhs vector
    ///////////////////////////////////////////////////////////////////////////////

    MPI_Datatype value_type = SFEM_EXPORT_FP32? MPI_FLOAT : MPI_DOUBLE;
    
    if(SFEM_EXPORT_FP32) {
        array_dtof(nnz,    (const double *)values, (float*)values);
        array_dtof(nnodes, (const double *)rhs, (float*)rhs);
    }

    {
        crs_t crs_out;
        crs_out.rowptr = (char *)rowptr;
        crs_out.colidx = (char *)colidx;
        crs_out.values = (char *)values;
        crs_out.grows = nnodes;
        crs_out.lrows = nnodes;
        crs_out.lnnz = nnz;
        crs_out.gnnz = nnz;
        crs_out.start = 0;
        crs_out.rowoffset = 0;
        crs_out.rowptr_type = MPI_INT;
        crs_out.colidx_type = MPI_INT;
        crs_out.values_type = value_type;
        crs_write_folder(comm, output_folder, &crs_out);
    }

    {
        sprintf(path, "%s/rhs.raw", output_folder);
        array_write(comm, path, value_type, rhs, nnodes, nnodes);
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
    free(rhs);

    for (int d = 0; d < 3; ++d) {
        free(xyz[d]);
    }

    for (int i = 0; i < 4; ++i) {
        free(elems[i]);
    }

    tock = MPI_Wtime();

    if (!rank) {
        printf("----------------------------------------\n");
        printf("#elements %ld #nodes %ld\n", (long)nelements, (long)nnodes);
        printf("TTS:\t\t\t%g seconds\n", tock - tick);
    }

    return MPI_Finalize();
}
