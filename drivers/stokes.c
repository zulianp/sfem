#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include "matrix.io/array_dtof.h"
#include "matrix.io/matrixio_array.h"
#include "matrix.io/matrixio_crs.h"
#include "matrix.io/utils.h"

#include "crs_graph.h"
#include "sfem_base.h"
#include "sfem_defs.h"
#include "sfem_vec.h"
#include "sortreduce.h"

#include "mass.h"

#include "dirichlet.h"
#include "neumann.h"

#include "read_mesh.h"

#include "stokes_mini.h"

SFEM_INLINE void tri3_stokes_mini_assemble_rhs_kernel(const real_t mu,
                                                      const real_t rho,
                                                      const real_t px0,
                                                      const real_t px1,
                                                      const real_t px2,
                                                      const real_t py0,
                                                      const real_t py1,
                                                      const real_t py2,
                                                      const real_t *const SFEM_RESTRICT u_rhs,
                                                      const real_t *const SFEM_RESTRICT p_rhs,
                                                      real_t *const SFEM_RESTRICT element_vector) {
    // const real_t x0 = 5 * u_rhs[2];
    // const real_t x1 = 9 * u_rhs[0];
    // const real_t x2 = 5 * u_rhs[3] + x1;
    // const real_t x3 = px0 - px1;
    // const real_t x4 = -py2;
    // const real_t x5 = py0 + x4;
    // const real_t x6 = -px2;
    // const real_t x7 = px0 + x6;
    // const real_t x8 = py0 - py1;
    // const real_t x9 = x3 * x5 - x7 * x8;
    // const real_t x10 = rho * x9;
    // const real_t x11 = (1.0 / 120.0) * x10;
    // const real_t x12 = 5 * u_rhs[1];
    // const real_t x13 = 5 * u_rhs[6];
    // const real_t x14 = 9 * u_rhs[4];
    // const real_t x15 = 5 * u_rhs[7] + x14;
    // const real_t x16 = 5 * u_rhs[5];
    // const real_t x17 = x9 * (27 * u_rhs[0] + 14 * u_rhs[1] + 14 * u_rhs[2] + 14 * u_rhs[3]);
    // const real_t x18 = x9 * (27 * u_rhs[4] + 14 * u_rhs[5] + 14 * u_rhs[6] + 14 * u_rhs[7]);
    // const real_t x19 = pow(x3, 2) - x3 * x7 + pow(x5, 2) - x5 * x8 + pow(x7, 2) + pow(x8, 2);
    // const real_t x20 = 140 * mu * x19;
    // const real_t x21 = (1.0 / 3360.0) * x10 / (mu * x19);
    // element_vector[0] = x11 * (10 * u_rhs[1] + x0 + x2);
    // element_vector[1] = x11 * (10 * u_rhs[2] + x12 + x2);
    // element_vector[2] = x11 * (10 * u_rhs[3] + x0 + x1 + x12);
    // element_vector[3] = x11 * (10 * u_rhs[5] + x13 + x15);
    // element_vector[4] = x11 * (10 * u_rhs[6] + x15 + x16);
    // element_vector[5] = x11 * (10 * u_rhs[7] + x13 + x14 + x16);
    // element_vector[6] =
    //     x21 * (x17 * (-py1 - x4) - x18 * (-px1 - x6) + x20 * (2 * p_rhs[0] + p_rhs[1] +
    //     p_rhs[2]));
    // element_vector[7] = x21 * (x17 * x5 - x18 * x7 + x20 * (p_rhs[0] + 2 * p_rhs[1] + p_rhs[2]));
    // element_vector[8] = x21 * (-x17 * x8 + x18 * x3 + x20 * (p_rhs[0] + p_rhs[1] + 2 *
    // p_rhs[2]));
    const real_t x0 = u_rhs[2] + u_rhs[3];
    const real_t x1 = px0 - px1;
    const real_t x2 = -py2;
    const real_t x3 = py0 + x2;
    const real_t x4 = -px2;
    const real_t x5 = px0 + x4;
    const real_t x6 = py0 - py1;
    const real_t x7 = x1 * x3 - x5 * x6;
    const real_t x8 = rho * x7;
    const real_t x9 = (1.0 / 24.0) * x8;
    const real_t x10 = u_rhs[6] + u_rhs[7];
    const real_t x11 = x7 * (u_rhs[1] + x0);
    const real_t x12 = x7 * (u_rhs[5] + x10);
    const real_t x13 = pow(x1, 2) - x1 * x5 + pow(x3, 2) - x3 * x6 + pow(x5, 2) + pow(x6, 2);
    const real_t x14 = 10 * mu * x13;
    const real_t x15 = (1.0 / 240.0) * x8 / (mu * x13);
    element_vector[0] = x9 * (2 * u_rhs[1] + x0);
    element_vector[1] = x9 * (u_rhs[1] + 2 * u_rhs[2] + u_rhs[3]);
    element_vector[2] = x9 * (u_rhs[1] + u_rhs[2] + 2 * u_rhs[3]);
    element_vector[3] = x9 * (2 * u_rhs[5] + x10);
    element_vector[4] = x9 * (u_rhs[5] + 2 * u_rhs[6] + u_rhs[7]);
    element_vector[5] = x9 * (u_rhs[5] + u_rhs[6] + 2 * u_rhs[7]);
    element_vector[6] =
        x15 * (x11 * (-py1 - x2) - x12 * (-px1 - x4) + x14 * (2 * p_rhs[0] + p_rhs[1] + p_rhs[2]));
    element_vector[7] = x15 * (x11 * x3 - x12 * x5 + x14 * (p_rhs[0] + 2 * p_rhs[1] + p_rhs[2]));
    element_vector[8] = x15 * (x1 * x12 - x11 * x6 + x14 * (p_rhs[0] + p_rhs[1] + 2 * p_rhs[2]));
}

// The MINI mixed finite element for the Stokes problem: An experimental
// investigation
static SFEM_INLINE real_t rhs1_x(const real_t mu, const real_t x, const real_t y) {
    return -mu * (4 * y * (1 - y) * (2 * y - 1) * ((1 - 2 * x) * (1 - 2 * x) - 2 * x * (1 - x)) +
                  12 * x * x * (1 - x) * (1 - x) * (1 - 2 * y)) +
           (1 - 2 * x) * (1 - y);
}
static SFEM_INLINE real_t rhs1_y(const real_t mu, const real_t x, const real_t y) {
    return -mu * (4 * x * (1 - x) * (1 - 2 * x) * ((1 - 2 * y) * (1 - 2 * y) - 2 * y * (1 - y)) +
                  12 * y * y * (1 - y) * (1 - y) * (2 * x - 1)) -
           x * (1 - x);
}

static SFEM_INLINE real_t rhs2_x(const real_t mu, const real_t x, const real_t y) {
    return -mu * ((2 - 12 * x + 12 * x * x) * (2 * y - 6 * y * y + 4 * y * y * y) +
                  (x * x - 2 * x * x * x + x * x * x * x) * (-12 + 24 * y)) +
           1. / 24;
}

static SFEM_INLINE real_t rhs2_y(const real_t mu, const real_t x, const real_t y) {
    return mu * ((2 - 12 * y + 12 * y * y) * (2 * x - 6 * x * x + 4 * x * x * x) +
                 (y * y - 2 * y * y * y + y * y * y * y) * (-12 + 24 * x)) +
           1. / 24;
}

static SFEM_INLINE real_t rhs3_x(const real_t mu, const real_t x, const real_t y) {
    const real_t pis4 = 4 * M_PI * M_PI;
    const real_t pi2 = 2 * M_PI;
    return -pis4 * mu * sin(pi2 * y) * (2 * cos(pi2 * x) - 1) + pis4 * sin(pi2 * x);
}

static SFEM_INLINE real_t rhs3_y(const real_t mu, const real_t x, const real_t y) {
    const real_t pis4 = 4 * M_PI * M_PI;
    const real_t pi2 = 2 * M_PI;
    return pis4 * mu * sin(pi2 * x) * (2 * cos(pi2 * y) - 1) - pis4 * sin(pi2 * y);
}

void node_eval_f2D(const ptrdiff_t nnodes,
                   geom_t **const points,
                   const real_t mu,
                   real_t (*f)(const real_t, const real_t, const real_t),
                   real_t *values) {
    for (ptrdiff_t i = 0; i < nnodes; i++) {
        values[i] = f(mu, points[0][i], points[1][i]);
        // printf
    }
}

//////////////////////////////////////////////

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

    struct stat st = {0};
    if (stat(output_folder, &st) == -1) {
        mkdir(output_folder, 0700);
    }

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

    // Optional params
    real_t SFEM_MU = 1;
    real_t SFEM_RHO = 1;
    int SFEM_PROBLEM_TYPE = 1;
    int SFEM_AOS = 0;
    const char *SFEM_DIRICHLET_NODES = 0;

    SFEM_READ_ENV(SFEM_PROBLEM_TYPE, atoi);
    SFEM_READ_ENV(SFEM_MU, atof);
    SFEM_READ_ENV(SFEM_RHO, atof);
    SFEM_READ_ENV(SFEM_AOS, atoi);
    SFEM_READ_ENV(SFEM_DIRICHLET_NODES, );

    if (rank == 0) {
        printf(
            "----------------------------------------\n"
            "Options:\n"
            "----------------------------------------\n"
            "- SFEM_PROBLEM_TYPE=%d\n"
            "- SFEM_MU=%g\n"
            "- SFEM_RHO=%g\n"
            "- SFEM_DIRICHLET_NODES=%s\n"
            "----------------------------------------\n",
            SFEM_PROBLEM_TYPE,
            SFEM_MU,
            SFEM_RHO,
            SFEM_DIRICHLET_NODES);
    }

    double tack = MPI_Wtime();
    printf("stokes.c: read\t\t%g seconds\n", tack - tick);

    ///////////////////////////////////////////////////////////////////////////////
    // Build CRS graph
    ///////////////////////////////////////////////////////////////////////////////

    ptrdiff_t nnz = 0;
    count_t *rowptr = 0;
    idx_t *colidx = 0;

    build_crs_graph_for_elem_type(
        mesh.element_type, mesh.nelements, mesh.nnodes, mesh.elements, &rowptr, &colidx);
    nnz = rowptr[mesh.nnodes];

    double tock = MPI_Wtime();
    printf("stokes.c: build crs graph\t\t%g seconds\n", tock - tack);
    tack = tock;

    const int sdim = elem_manifold_dim(mesh.element_type);
    const int n_vars = sdim + 1;

    real_t *rhs_values[4] = {0, 0, 0, 0};
    switch (SFEM_PROBLEM_TYPE) {
        case 1: {
            rhs_values[0] = calloc(mesh.nnodes, sizeof(real_t));
            rhs_values[1] = calloc(mesh.nnodes, sizeof(real_t));
            node_eval_f2D(mesh.nnodes, mesh.points, SFEM_MU, &rhs1_x, rhs_values[0]);
            node_eval_f2D(mesh.nnodes, mesh.points, SFEM_MU, &rhs1_y, rhs_values[1]);
            break;
        }
        case 2: {
            rhs_values[0] = calloc(mesh.nnodes, sizeof(real_t));
            rhs_values[1] = calloc(mesh.nnodes, sizeof(real_t));
            node_eval_f2D(mesh.nnodes, mesh.points, SFEM_MU, &rhs2_x, rhs_values[0]);
            node_eval_f2D(mesh.nnodes, mesh.points, SFEM_MU, &rhs2_y, rhs_values[1]);
            break;
        }
        case 3: {
            rhs_values[0] = calloc(mesh.nnodes, sizeof(real_t));
            rhs_values[1] = calloc(mesh.nnodes, sizeof(real_t));
            node_eval_f2D(mesh.nnodes, mesh.points, SFEM_MU, &rhs3_x, rhs_values[0]);
            node_eval_f2D(mesh.nnodes, mesh.points, SFEM_MU, &rhs3_y, rhs_values[1]);
            break;
        }
        default:
            break;
    }

    if (SFEM_AOS) {
        real_t *values = calloc(n_vars * n_vars * nnz, sizeof(real_t));
        real_t *rhs = calloc(n_vars * mesh.nnodes, sizeof(real_t));

        ///////////////////////////////////////////////////////////////////////////////
        // Operator assembly
        ///////////////////////////////////////////////////////////////////////////////

        stokes_mini_assemble_hessian_aos(mesh.element_type,
                                         mesh.nelements,
                                         mesh.nnodes,
                                         mesh.elements,
                                         mesh.points,
                                         SFEM_MU,
                                         rowptr,
                                         colidx,
                                         values);

        stokes_mini_assemble_rhs_aos(mesh.element_type,
                                     mesh.nelements,
                                     mesh.nnodes,
                                     mesh.elements,
                                     mesh.points,
                                     SFEM_MU,
                                     SFEM_RHO,
                                     rhs_values,
                                     rhs);

        count_t *b_rowptr = (count_t *)malloc((mesh.nnodes + 1) * n_vars * sizeof(count_t));
        idx_t *b_colidx = (idx_t *)malloc(rowptr[mesh.nnodes] * n_vars * n_vars * sizeof(idx_t));
        crs_graph_block_to_scalar(mesh.nnodes, n_vars, rowptr, colidx, b_rowptr, b_colidx);

        if (SFEM_DIRICHLET_NODES) {
            idx_t *dirichlet_nodes = 0;
            ptrdiff_t _nope_, nn;
            array_create_from_file(comm,
                                   SFEM_DIRICHLET_NODES,
                                   SFEM_MPI_IDX_T,
                                   (void **)&dirichlet_nodes,
                                   &_nope_,
                                   &nn);

            for (int d = 0; d < sdim; d++) {
                constraint_nodes_to_value_vec(nn, dirichlet_nodes, n_vars, d, 0, rhs);
            }

            for (int d1 = 0; d1 < sdim; d1++) {
                crs_constraint_nodes_to_identity_vec(
                    nn, dirichlet_nodes, n_vars, d1, 1, b_rowptr, b_colidx, values);
            }

            if (1)
            // if (0)
            {
                // One point to 0 to fix pressure degree of freedom
                // ptrdiff_t node = nn - 1;
                ptrdiff_t node = 0;
                crs_constraint_nodes_to_identity_vec(
                    1, &dirichlet_nodes[node], n_vars, (n_vars - 1), 1, b_rowptr, b_colidx, values);

                constraint_nodes_to_value_vec(
                    1, &dirichlet_nodes[node], n_vars, n_vars - 1, 0, rhs);
            }

        } else {
            assert(0);
        }

        {
            crs_t crs_out;
            crs_out.rowptr = (char *)b_rowptr;
            crs_out.colidx = (char *)b_colidx;
            crs_out.values = (char *)values;
            crs_out.grows = mesh.nnodes * n_vars;
            crs_out.lrows = mesh.nnodes * n_vars;
            crs_out.lnnz = b_rowptr[mesh.nnodes * n_vars];
            crs_out.gnnz = b_rowptr[mesh.nnodes * n_vars];
            crs_out.start = 0;
            crs_out.rowoffset = 0;
            crs_out.rowptr_type = SFEM_MPI_COUNT_T;
            crs_out.colidx_type = SFEM_MPI_IDX_T;
            crs_out.values_type = SFEM_MPI_REAL_T;

            crs_write_folder(comm, output_folder, &crs_out);
        }

        {
            char path[1024 * 10];
            // Write rhs vectors
            sprintf(path, "%s/rhs.raw", output_folder);
            array_write(
                comm, path, SFEM_MPI_REAL_T, rhs, mesh.nnodes * n_vars, mesh.nnodes * n_vars);
        }

        free(b_rowptr);
        free(b_colidx);
        free(values);
        free(rhs);
    } else {
        real_t **values = 0;
        values = (real_t **)malloc((n_vars * n_vars) * sizeof(real_t *));
        for (int d = 0; d < (n_vars * n_vars); d++) {
            values[d] = calloc(nnz, sizeof(real_t));
        }

        real_t **rhs = 0;
        rhs = (real_t **)malloc((n_vars) * sizeof(real_t *));
        for (int d = 0; d < n_vars; d++) {
            rhs[d] = calloc(mesh.nnodes, sizeof(real_t));
        }

        ///////////////////////////////////////////////////////////////////////////////
        // Operator assembly
        ///////////////////////////////////////////////////////////////////////////////

        stokes_mini_assemble_hessian_soa(mesh.element_type,
                                         mesh.nelements,
                                         mesh.nnodes,
                                         mesh.elements,
                                         mesh.points,
                                         SFEM_MU,
                                         rowptr,
                                         colidx,
                                         values);

        stokes_mini_assemble_rhs_soa(mesh.element_type,
                                     mesh.nelements,
                                     mesh.nnodes,
                                     mesh.elements,
                                     mesh.points,
                                     SFEM_MU,
                                     SFEM_RHO,
                                     rhs_values,
                                     rhs);

        tock = MPI_Wtime();
        printf("stokes.c: assembly\t\t%g seconds\n", tock - tack);
        tack = tock;

        ///////////////////////////////////////////////////////////////////////////////
        // Boundary conditions
        ///////////////////////////////////////////////////////////////////////////////

        if (SFEM_DIRICHLET_NODES) {
            idx_t *dirichlet_nodes = 0;
            ptrdiff_t _nope_, nn;
            array_create_from_file(comm,
                                   SFEM_DIRICHLET_NODES,
                                   SFEM_MPI_IDX_T,
                                   (void **)&dirichlet_nodes,
                                   &_nope_,
                                   &nn);

            for (int d = 0; d < sdim; d++) {
                constraint_nodes_to_value(nn, dirichlet_nodes, 0, rhs[d]);
            }

            for (int d1 = 0; d1 < sdim; d1++) {
                for (int d2 = 0; d2 < n_vars; d2++) {
                    crs_constraint_nodes_to_identity(
                        nn, dirichlet_nodes, d1 == d2, rowptr, colidx, values[d1 * n_vars + d2]);
                }
            }

            if (1)
            // if (0)
            {
                // One point to 0 to fix pressure degree of freedom
                // ptrdiff_t node = nn - 1;
                ptrdiff_t node = 0;
                for (int d2 = 0; d2 < n_vars; d2++) {
                    crs_constraint_nodes_to_identity(1,
                                                     &dirichlet_nodes[node],
                                                     (n_vars - 1) == d2,
                                                     rowptr,
                                                     colidx,
                                                     values[(n_vars - 1) * n_vars + d2]);
                }

                constraint_nodes_to_value(1, &dirichlet_nodes[node], 0, rhs[n_vars - 1]);
            }

        } else {
            assert(0);
        }

        tock = MPI_Wtime();
        printf("stokes.c: boundary\t\t%g seconds\n", tock - tack);
        tack = tock;

        ///////////////////////////////////////////////////////////////////////////////
        // Write to disk
        ///////////////////////////////////////////////////////////////////////////////

        {
            // Write block CRS matrix
            block_crs_t crs_out;
            crs_out.rowptr = (char *)rowptr;
            crs_out.colidx = (char *)colidx;

            crs_out.block_size = n_vars * n_vars;
            crs_out.values = (char **)values;
            crs_out.grows = mesh.nnodes;
            crs_out.lrows = mesh.nnodes;
            crs_out.lnnz = nnz;
            crs_out.gnnz = nnz;
            crs_out.start = 0;
            crs_out.rowoffset = 0;
            crs_out.rowptr_type = SFEM_MPI_COUNT_T;
            crs_out.colidx_type = SFEM_MPI_IDX_T;
            crs_out.values_type = SFEM_MPI_REAL_T;

            char path_rowptr[1024 * 10];
            sprintf(path_rowptr, "%s/rowptr.raw", output_folder);

            char path_colidx[1024 * 10];
            sprintf(path_colidx, "%s/colidx.raw", output_folder);

            char format_values[1024 * 10];
            sprintf(format_values, "%s/values.%%d.raw", output_folder);
            block_crs_write(comm, path_rowptr, path_colidx, format_values, &crs_out);
        }

        {
            char path[1024 * 10];
            // Write rhs vectors
            for (int d = 0; d < n_vars; d++) {
                sprintf(path, "%s/rhs.%d.raw", output_folder, d);
                array_write(comm, path, SFEM_MPI_REAL_T, rhs[d], mesh.nnodes, mesh.nnodes);
            }
        }

        tock = MPI_Wtime();
        printf("stokes.c: write\t\t%g seconds\n", tock - tack);
        tack = tock;

        ///////////////////////////////////////////////////////////////////////////////
        // Free resources
        ///////////////////////////////////////////////////////////////////////////////

        for (int d = 0; d < (n_vars * n_vars); d++) {
            free(values[d]);
        }

        free(values);

        for (int d = 0; d < n_vars; d++) {
            free(rhs[d]);
        }

        free(rhs);

        for (int d = 0; d < n_vars; d++) {
            if (rhs_values[d]) {
                free(rhs_values[d]);
            }
        }
    }

    // Mesh n2n graph
    free(rowptr);
    free(colidx);

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
