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

#include "tet4_laplacian.h"
#include "cvfem_tri3_diffusion.h"

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
        fprintf(stderr, "usage: %s <folder> <output_folder>\n", argv[0]);
        return EXIT_FAILURE;
    }

    const char *output_folder = argv[2];

    printf("%s %s %s\n", argv[0], argv[1], output_folder);

    int SFEM_LAPLACIAN = 1;
    int SFEM_HANDLE_DIRICHLET = 1;
    int SFEM_HANDLE_NEUMANN = 0;
    int SFEM_HANDLE_RHS = 0;
    int SFEM_EXPORT_FP32 = 0;

    SFEM_READ_ENV(SFEM_LAPLACIAN, atoi);
    SFEM_READ_ENV(SFEM_HANDLE_DIRICHLET, atoi);
    SFEM_READ_ENV(SFEM_EXPORT_FP32, atoi);
    SFEM_READ_ENV(SFEM_HANDLE_NEUMANN, atoi);
    SFEM_READ_ENV(SFEM_HANDLE_RHS, atoi);

    printf("----------------------------------------\n");
    printf(
        "Environment variables:\n- SFEM_LAPLACIAN=%d\n- SFEM_HANDLE_DIRICHLET=%d\n- "
        "SFEM_HANDLE_NEUMANN=%d\n- SFEM_HANDLE_RHS=%d\n- SFEM_EXPORT_FP32=%d\n",
        SFEM_LAPLACIAN,
        SFEM_HANDLE_DIRICHLET,
        SFEM_HANDLE_NEUMANN,
        SFEM_HANDLE_RHS,
        SFEM_EXPORT_FP32);
    printf("----------------------------------------\n");

    MPI_Datatype value_type = SFEM_EXPORT_FP32 ? MPI_FLOAT : MPI_DOUBLE;

    double tick = MPI_Wtime();

    ///////////////////////////////////////////////////////////////////////////////
    // Read data
    ///////////////////////////////////////////////////////////////////////////////

    const char *folder = argv[1];

    mesh_t mesh;
    if (mesh_read(comm, folder, &mesh)) {
        return EXIT_FAILURE;
    }

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
        mesh.element_type, mesh.nelements, mesh.nnodes, mesh.elements, &rowptr, &colidx);

    nnz = rowptr[mesh.nnodes];
    values = (real_t *)malloc(nnz * sizeof(real_t));
    memset(values, 0, nnz * sizeof(real_t));

    double tock = MPI_Wtime();
    printf("assemble.c: build crs\t\t%g seconds\n", tock - tack);
    tack = tock;

    ///////////////////////////////////////////////////////////////////////////////
    // Operator assembly
    ///////////////////////////////////////////////////////////////////////////////
    if (SFEM_LAPLACIAN) {
        switch (mesh.element_type) {
            case TRI3: {
                cvfem_tri3_diffusion_assemble_hessian(mesh.nelements,
                                                      mesh.nnodes,
                                                      mesh.elements,
                                                      mesh.points,
                                                      rowptr,
                                                      colidx,
                                                      values);
                break;
            }
            case TET4: {
                tet4_laplacian_assemble_hessian(mesh.nelements,
                                                      mesh.nnodes,
                                                      mesh.elements,
                                                      mesh.points,
                                                      rowptr,
                                                      colidx,
                                                      values);
                break;
            }
            default:
                MPI_Finalize();
                return EXIT_FAILURE;
        }
    }

    tock = MPI_Wtime();
    printf("assemble.c: assembly\t\t%g seconds\n", tock - tack);
    tack = tock;

    ///////////////////////////////////////////////////////////////////////////////
    // Boundary conditions
    ///////////////////////////////////////////////////////////////////////////////

    real_t *rhs = (real_t *)malloc(mesh.nnodes * sizeof(real_t));
    memset(rhs, 0, mesh.nnodes * sizeof(real_t));

    if (SFEM_HANDLE_NEUMANN) {  // Neumann
        char path[1024 * 10];
        sprintf(path, "%s/on.raw", folder);

        const char *SFEM_NEUMANN_FACES = 0;
        SFEM_READ_ENV(SFEM_NEUMANN_FACES, );

        if (SFEM_NEUMANN_FACES) {
            strcpy(path, SFEM_NEUMANN_FACES);
            printf("SFEM_NEUMANN_FACES=%s\n", path);
        }

        idx_t *faces_neumann = 0;

        enum ElemType st = shell_type(side_type(mesh.element_type));
        int nnodesxface = elem_num_nodes(st);
        ptrdiff_t nfacesxnxe = read_file(comm, path, (void **)&faces_neumann);
        idx_t nfaces = (nfacesxnxe / nnodesxface) / sizeof(idx_t);
        assert(nfaces * nnodesxface * sizeof(idx_t) == nfacesxnxe);

        surface_forcing_function(st, nfaces, faces_neumann, mesh.points, 1.0, rhs);
        free(faces_neumann);
    }

    if (SFEM_HANDLE_DIRICHLET) {
        // Dirichlet
        char path[1024 * 10];
        sprintf(path, "%s/zd.raw", folder);

        const char *SFEM_DIRICHLET_NODES = 0;
        SFEM_READ_ENV(SFEM_DIRICHLET_NODES, );

        if (SFEM_DIRICHLET_NODES) {
            strcpy(path, SFEM_DIRICHLET_NODES);
            printf("SFEM_DIRICHLET_NODES=%s\n", path);
        }

        idx_t *dirichlet_nodes = 0;
        ptrdiff_t nn = read_file(comm, path, (void **)&dirichlet_nodes);
        assert((nn / sizeof(idx_t)) * sizeof(idx_t) == nn);
        nn /= sizeof(idx_t);

        constraint_nodes_to_value(nn, dirichlet_nodes, 0, rhs);
        crs_constraint_nodes_to_identity(nn, dirichlet_nodes, 1, rowptr, colidx, values);
    }

    if (SFEM_HANDLE_RHS) {
        if (SFEM_EXPORT_FP32) {
            array_dtof(mesh.nnodes, (const real_t *)rhs, (float *)rhs);
        }

        {
            char path[1024 * 10];
            sprintf(path, "%s/rhs.raw", output_folder);
            array_write(comm, path, value_type, rhs, mesh.nnodes, mesh.nnodes);
        }
    }

    free(rhs);

    tock = MPI_Wtime();
    printf("assemble.c: boundary\t\t%g seconds\n", tock - tack);
    tack = tock;

    ///////////////////////////////////////////////////////////////////////////////
    // Write CRS matrix and rhs vector
    ///////////////////////////////////////////////////////////////////////////////

    if (SFEM_EXPORT_FP32) {
        array_dtof(nnz, (const real_t *)values, (float *)values);
    }

    {
        crs_t crs_out;
        crs_out.rowptr = (char *)rowptr;
        crs_out.colidx = (char *)colidx;
        crs_out.values = (char *)values;
        crs_out.grows = mesh.nnodes;
        crs_out.lrows = mesh.nnodes;
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

    ptrdiff_t nelements = mesh.nelements;
    ptrdiff_t nnodes = mesh.nnodes;

    mesh_destroy(&mesh);

    tock = MPI_Wtime();

    if (!rank) {
        printf("----------------------------------------\n");
        printf("#elements %ld #nodes %ld #nz %ld\n", (long)nelements, (long)nnodes, (long)nnz);
        printf("TTS:\t\t\t%g seconds\n", tock - tick);
    }

    return MPI_Finalize();
}
