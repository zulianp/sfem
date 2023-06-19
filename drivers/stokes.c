#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../matrix.io/array_dtof.h"
#include "../matrix.io/matrixio_array.h"
#include "../matrix.io/matrixio_crs.h"
#include "../matrix.io/utils.h"

#include "crs_graph.h"
#include "sfem_base.h"
#include "sfem_defs.h"
#include "sfem_vec.h"
#include "sortreduce.h"

#include "mass.h"

#include "dirichlet.h"
#include "neumann.h"

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

static SFEM_INLINE int linear_search(const idx_t target, const idx_t *const arr, const int size) {
    int i;
    for (i = 0; i < size - SFEM_VECTOR_SIZE; i += SFEM_VECTOR_SIZE) {
        if (arr[i] == target) return i;
        if (arr[i + 1] == target) return i + 1;
        if (arr[i + 2] == target) return i + 2;
        if (arr[i + 3] == target) return i + 3;
    }
    for (; i < size; i++) {
        if (arr[i] == target) return i;
    }
    return -1;
}

static SFEM_INLINE int find_col(const idx_t key, const idx_t *const row, const int lenrow) {
    if (lenrow <= 32) {
        return linear_search(key, row, lenrow);

        // Using sentinel (potentially dangerous if matrix is buggy and column does not exist)
        // while (key > row[++k]) {
        //     // Hi
        // }
        // assert(k < lenrow);
        // assert(key == row[k]);
    } else {
        // Use this for larger number of dofs per row
        return find_idx_binary_search(key, row, lenrow);
    }
}

static SFEM_INLINE void find_cols3(const idx_t *targets,
                                   const idx_t *const row,
                                   const int lenrow,
                                   int *ks) {
    if (lenrow > 32) {
        for (int d = 0; d < 3; ++d) {
            ks[d] = find_col(targets[d], row, lenrow);
        }
    } else {
#pragma unroll(3)
        for (int d = 0; d < 3; ++d) {
            ks[d] = 0;
        }

        for (int i = 0; i < lenrow; ++i) {
#pragma unroll(3)
            for (int d = 0; d < 3; ++d) {
                ks[d] += row[i] < targets[d];
            }
        }
    }
}

SFEM_INLINE void tri3_stokes_assemble_hessian_kernel(const real_t mu,
                                                     const real_t px0,
                                                     const real_t px1,
                                                     const real_t px2,
                                                     const real_t py0,
                                                     const real_t py1,
                                                     const real_t py2,
                                                     real_t *const SFEM_RESTRICT element_matrix) {
    const real_t x0 = px0 - px2;
    const real_t x1 = px0 - px1;
    const real_t x2 = -x1;
    const real_t x3 = py0 - py2;
    const real_t x4 = -x3;
    const real_t x5 = py0 - py1;
    const real_t x6 = -x0 * x5 + x2 * x4;
    const real_t x7 = pow(x6, -2);
    const real_t x8 = pow(x0, 2) * x7 + pow(x4, 2) * x7;
    const real_t x9 = -x0 * x5 + x1 * x3;
    const real_t x10 = mu * x9;
    const real_t x11 = x10 * x8;
    const real_t x12 = (1.0 / 2.0) * x11;
    const real_t x13 = -x12;
    const real_t x14 = x0 * x2 * x7 + x4 * x5 * x7;
    const real_t x15 = x10 * x14;
    const real_t x16 = (1.0 / 2.0) * x15;
    const real_t x17 = -x16;
    const real_t x18 = 1.0 / x6;
    const real_t x19 = x0 * x18;
    const real_t x20 = x18 * x4;
    const real_t x21 = x19 + x20;
    const real_t x22 = (1.0 / 6.0) * x9;
    const real_t x23 = x21 * x22;
    const real_t x24 = -x20 * x22;
    const real_t x25 = -x19 * x22;
    const real_t x26 = pow(x2, 2) * x7 + pow(x5, 2) * x7;
    const real_t x27 = x10 * x26;
    const real_t x28 = (1.0 / 2.0) * x27;
    const real_t x29 = -x28;
    const real_t x30 = x18 * x2;
    const real_t x31 = x18 * x5;
    const real_t x32 = x30 + x31;
    const real_t x33 = x22 * x32;
    const real_t x34 = -x22 * x31;
    const real_t x35 = -x22 * x30;
    const real_t x36 = pow(x9, 2);
    const real_t x37 = (6561.0 / 100.0) * pow(mu, 2) * x36;
    const real_t x38 = 1.0 / (-pow(x14, 2) * x37 + x26 * x37 * x8);
    const real_t x39 = (729.0 / 400.0) * mu * x36 * x38;
    const real_t x40 = x32 * x39;
    const real_t x41 = (729.0 / 400.0) * mu * x14 * x21 * x36 * x38 - x40 * x8;
    const real_t x42 = (9.0 / 40.0) * x9;
    const real_t x43 = x32 * x42;
    const real_t x44 = x14 * x40 - x21 * x26 * x39;
    const real_t x45 = x21 * x42;
    const real_t x46 = x42 * x44;
    const real_t x47 = x41 * x42;
    const real_t x48 = x14 * x39;
    const real_t x49 = (729.0 / 400.0) * mu * x18 * x36 * x38 * x5 * x8 - x20 * x48;
    const real_t x50 = x20 * x26 * x39 - x31 * x48;
    const real_t x51 = x42 * x50;
    const real_t x52 = x42 * x49;
    const real_t x53 = -x19 * x48 + x30 * x39 * x8;
    const real_t x54 = (729.0 / 400.0) * mu * x0 * x18 * x26 * x36 * x38 - x30 * x48;
    const real_t x55 = x42 * x54;
    const real_t x56 = x42 * x53;
    element_matrix[0] = x11;
    element_matrix[1] = x13;
    element_matrix[2] = x13;
    element_matrix[3] = x15;
    element_matrix[4] = x17;
    element_matrix[5] = x17;
    element_matrix[6] = x23;
    element_matrix[7] = x23;
    element_matrix[8] = x23;
    element_matrix[9] = x13;
    element_matrix[10] = x12;
    element_matrix[11] = 0;
    element_matrix[12] = x17;
    element_matrix[13] = x16;
    element_matrix[14] = 0;
    element_matrix[15] = x24;
    element_matrix[16] = x24;
    element_matrix[17] = x24;
    element_matrix[18] = x13;
    element_matrix[19] = 0;
    element_matrix[20] = x12;
    element_matrix[21] = x17;
    element_matrix[22] = 0;
    element_matrix[23] = x16;
    element_matrix[24] = x25;
    element_matrix[25] = x25;
    element_matrix[26] = x25;
    element_matrix[27] = x15;
    element_matrix[28] = x17;
    element_matrix[29] = x17;
    element_matrix[30] = x27;
    element_matrix[31] = x29;
    element_matrix[32] = x29;
    element_matrix[33] = x33;
    element_matrix[34] = x33;
    element_matrix[35] = x33;
    element_matrix[36] = x17;
    element_matrix[37] = x16;
    element_matrix[38] = 0;
    element_matrix[39] = x29;
    element_matrix[40] = x28;
    element_matrix[41] = 0;
    element_matrix[42] = x34;
    element_matrix[43] = x34;
    element_matrix[44] = x34;
    element_matrix[45] = x17;
    element_matrix[46] = 0;
    element_matrix[47] = x16;
    element_matrix[48] = x29;
    element_matrix[49] = 0;
    element_matrix[50] = x28;
    element_matrix[51] = x35;
    element_matrix[52] = x35;
    element_matrix[53] = x35;
    element_matrix[54] = x23;
    element_matrix[55] = x24;
    element_matrix[56] = x25;
    element_matrix[57] = x33;
    element_matrix[58] = x34;
    element_matrix[59] = x35;
    element_matrix[60] = x41 * x43 + x44 * x45;
    element_matrix[61] = -x20 * x46 - x31 * x47;
    element_matrix[62] = -x19 * x46 - x30 * x47;
    element_matrix[63] = x23;
    element_matrix[64] = x24;
    element_matrix[65] = x25;
    element_matrix[66] = x33;
    element_matrix[67] = x34;
    element_matrix[68] = x35;
    element_matrix[69] = x43 * x49 + x45 * x50;
    element_matrix[70] = -x20 * x51 - x31 * x52;
    element_matrix[71] = -x19 * x51 - x30 * x52;
    element_matrix[72] = x23;
    element_matrix[73] = x24;
    element_matrix[74] = x25;
    element_matrix[75] = x33;
    element_matrix[76] = x34;
    element_matrix[77] = x35;
    element_matrix[78] = x43 * x53 + x45 * x54;
    element_matrix[79] = -x20 * x55 - x31 * x56;
    element_matrix[80] = -x19 * x55 - x30 * x56;
}

SFEM_INLINE void tri3_mini_condense_rhs(const real_t mu,
                                        const real_t px0,
                                        const real_t px1,
                                        const real_t px2,
                                        const real_t py0,
                                        const real_t py1,
                                        const real_t py2,
                                        real_t *const SFEM_RESTRICT rhs_bubble,
                                        real_t *const SFEM_RESTRICT element_vector) {
    const real_t x0 = px0 - px1;
    const real_t x1 = -x0;
    const real_t x2 = py0 - py2;
    const real_t x3 = -x2;
    const real_t x4 = px0 - px2;
    const real_t x5 = py0 - py1;
    const real_t x6 = x1 * x3 - x4 * x5;
    const real_t x7 = pow(x6, -2);
    const real_t x8 = pow(x1, 2) * x7 + pow(x5, 2) * x7;
    const real_t x9 = 1.0 / x6;
    const real_t x10 = x4 * x9;
    const real_t x11 = x3 * x9;
    const real_t x12 = x10 + x11;
    const real_t x13 = pow(x0 * x2 - x4 * x5, 2);
    const real_t x14 = x1 * x4 * x7 + x3 * x5 * x7;
    const real_t x15 = (6561.0 / 100.0) * pow(mu, 2) * x13;
    const real_t x16 = pow(x3, 2) * x7 + pow(x4, 2) * x7;
    const real_t x17 = 1.0 / (-pow(x14, 2) * x15 + x15 * x16 * x8);
    const real_t x18 = (729.0 / 400.0) * mu * x13 * x17;
    const real_t x19 = x1 * x9;
    const real_t x20 = x5 * x9;
    const real_t x21 = x18 * (x19 + x20);
    const real_t x22 = x14 * x18;
    element_vector[0] = 0;
    element_vector[1] = 0;
    element_vector[2] = 0;
    element_vector[3] = 0;
    element_vector[4] = 0;
    element_vector[5] = 0;
    element_vector[6] = rhs_bubble[0] * (-x12 * x18 * x8 + x14 * x21) +
                        rhs_bubble[1] * ((729.0 / 400.0) * mu * x12 * x13 * x14 * x17 - x16 * x21);
    element_vector[7] =
        rhs_bubble[0] * (x11 * x18 * x8 - x20 * x22) +
        rhs_bubble[1] * ((729.0 / 400.0) * mu * x13 * x16 * x17 * x5 * x9 - x11 * x22);
    element_vector[8] =
        rhs_bubble[0] * ((729.0 / 400.0) * mu * x13 * x17 * x4 * x8 * x9 - x19 * x22) +
        rhs_bubble[1] * (-x10 * x22 + x16 * x18 * x19);
}

void tri3_stokes_mini_assemble_hessian(const real_t mu,
                                       enum ElemType element_type,
                                       const ptrdiff_t nelements,
                                       const ptrdiff_t nnodes,
                                       idx_t **const elems,
                                       geom_t **const points,
                                       const count_t *const rowptr,
                                       const idx_t *const colidx,
                                       real_t **const values) {
    assert(element_type == TRI3);
    SFEM_UNUSED(nnodes);
    SFEM_UNUSED(element_type);

    double tick = MPI_Wtime();

    static const int n_vars = 3;
    static const int ndofs = 3;
    static const int rows = 9;
    static const int cols = 9;

    idx_t ev[3];
    idx_t ks[3];
    real_t element_matrix[3 * 3 * 3 * 3];

    for (ptrdiff_t i = 0; i < nelements; ++i) {
#pragma unroll(3)
        for (int v = 0; v < 3; ++v) {
            ev[v] = elems[v][i];
        }

        // Element indices
        const idx_t i0 = ev[0];
        const idx_t i1 = ev[1];
        const idx_t i2 = ev[2];

        tri3_stokes_assemble_hessian_kernel(mu,
                                            // X-coordinates
                                            points[0][i0],
                                            points[0][i1],
                                            points[0][i2],
                                            // Y-coordinates
                                            points[1][i0],
                                            points[1][i1],
                                            points[1][i2],
                                            element_matrix);

        for (int edof_i = 0; edof_i < 3; ++edof_i) {
            const idx_t dof_i = elems[edof_i][i];
            const idx_t lenrow = rowptr[dof_i + 1] - rowptr[dof_i];

            const idx_t *row = &colidx[rowptr[dof_i]];

            // Find indices
            find_cols3(ev, row, lenrow, ks);

            // Add block
            for (int d1 = 0; d1 < n_vars; d1++) {
                for (int d2 = 0; d2 < n_vars; d2++) {
                    real_t *rowvalues = &(values[d1 * n_vars + d2][rowptr[dof_i]]);
                    int lidx_i = d1 * ndofs + edof_i;
                    const real_t *element_row = &element_matrix[lidx_i * cols];

#pragma unroll(3)
                    for (int edof_j = 0; edof_j < 3; ++edof_j) {
                        int lidx_j = d2 * ndofs + edof_j;
                        rowvalues[ks[edof_j]] += element_row[lidx_j];
                    }
                }
            }
        }
    }

    double tock = MPI_Wtime();
    printf("tri3_stokes.c: tri3_stokes_assemble_hessian\t%g seconds\n", tock - tick);
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

    if (argc != 2) {
        fprintf(stderr, "usage: %s <folder> <output>\n", argv[0]);
        return EXIT_FAILURE;
    }

    const char *output_folder = argv[2];

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

    double tack = MPI_Wtime();
    printf("stokes.c: read\t\t%g seconds\n", tack - tick);

    ///////////////////////////////////////////////////////////////////////////////
    // Build CRS graph
    ///////////////////////////////////////////////////////////////////////////////

    ptrdiff_t nnz = 0;
    count_t *rowptr = 0;
    idx_t *colidx = 0;
    real_t **values = 0;

    build_crs_graph_for_elem_type(
        mesh.element_type, mesh.nelements, mesh.nnodes, mesh.elements, &rowptr, &colidx);

    nnz = rowptr[mesh.nnodes];

    static const int n_vars = 3;
    values = (real_t **)malloc((n_vars * n_vars) * sizeof(real_t *));
    for (int d = 0; d < n_vars * n_vars; d++) {
        values[d] = calloc(nnz, sizeof(real_t));
    }

    double tock = MPI_Wtime();
    printf("stokes.c: build crs\t\t%g seconds\n", tock - tack);
    tack = tock;

    ///////////////////////////////////////////////////////////////////////////////
    // Operator assembly
    ///////////////////////////////////////////////////////////////////////////////

    real_t mu = 1;
    tri3_stokes_mini_assemble_hessian(mu,
                                      mesh.element_type,
                                      mesh.nelements,
                                      mesh.nnodes,
                                      mesh.elements,
                                      mesh.points,
                                      rowptr,
                                      colidx,
                                      values);

    tock = MPI_Wtime();
    printf("stokes.c: assembly\t\t%g seconds\n", tock - tack);
    tack = tock;

    ///////////////////////////////////////////////////////////////////////////////
    // Boundary conditions
    ///////////////////////////////////////////////////////////////////////////////

    // real_t *rhs = (real_t *)malloc(mesh.nnodes * sizeof(real_t));
    // memset(rhs, 0, mesh.nnodes * sizeof(real_t));

    // if (SFEM_HANDLE_NEUMANN) {  // Neumann
    //     char path[1024 * 10];
    //     sprintf(path, "%s/on.raw", folder);

    //     const char *SFEM_NEUMANN_FACES = 0;
    //     SFEM_READ_ENV(SFEM_NEUMANN_FACES, );

    //     if (SFEM_NEUMANN_FACES) {
    //         strcpy(path, SFEM_NEUMANN_FACES);
    //         printf("SFEM_NEUMANN_FACES=%s\n", path);
    //     }

    //     idx_t *faces_neumann = 0;

    //     enum ElemType st = side_type(mesh.element_type);
    //     int nnodesxface = elem_num_nodes(st);
    //     ptrdiff_t nfacesxnxe = read_file(comm, path, (void **)&faces_neumann);
    //     idx_t nfaces = (nfacesxnxe / nnodesxface) / sizeof(idx_t);
    //     assert(nfaces * nnodesxface * sizeof(idx_t) == nfacesxnxe);

    //     surface_forcing_function(st, nfaces, faces_neumann, mesh.points, 1.0, rhs);
    //     free(faces_neumann);
    // }

    // if (SFEM_HANDLE_DIRICHLET) {
    //     // Dirichlet
    //     char path[1024 * 10];
    //     sprintf(path, "%s/zd.raw", folder);

    //     const char *SFEM_DIRICHLET_NODES = 0;
    //     SFEM_READ_ENV(SFEM_DIRICHLET_NODES, );

    //     if (SFEM_DIRICHLET_NODES) {
    //         strcpy(path, SFEM_DIRICHLET_NODES);
    //         printf("SFEM_DIRICHLET_NODES=%s\n", path);
    //     }

    //     idx_t *dirichlet_nodes = 0;
    //     ptrdiff_t nn = read_file(comm, path, (void **)&dirichlet_nodes);
    //     assert((nn / sizeof(idx_t)) * sizeof(idx_t) == nn);
    //     nn /= sizeof(idx_t);

    //     constraint_nodes_to_value(nn, dirichlet_nodes, 0, rhs);
    //     crs_constraint_nodes_to_identity(nn, dirichlet_nodes, 1, rowptr, colidx, values);
    // }

    // if (SFEM_HANDLE_RHS) {
    //     if (SFEM_EXPORT_FP32) {
    //         array_dtof(mesh.nnodes, (const real_t *)rhs, (float *)rhs);
    //     }

    //     {
    //         char path[1024 * 10];
    //         sprintf(path, "%s/rhs.raw", output_folder);
    //         array_write(comm, path, value_type, rhs, mesh.nnodes, mesh.nnodes);
    //     }
    // }

    // free(rhs);

    tock = MPI_Wtime();
    printf("stokes.c: boundary\t\t%g seconds\n", tock - tack);
    tack = tock;

    tock = MPI_Wtime();
    printf("stokes.c: write\t\t%g seconds\n", tock - tack);
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
