#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

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
#include "stokes_mini.h"

////////////////////////////////////////////////////////////////////////////////////
// The MINI mixed finite element for the Stokes problem: An experimental
// investigation
////////////////////////////////////////////////////////////////////////////////////
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

static SFEM_INLINE real_t rhs4_x(const real_t mu, const real_t x, const real_t y, const real_t z) {
    const real_t pis4 = 4 * M_PI * M_PI;
    const real_t pi2 = 2 * M_PI;
    return -pis4 * mu * sin(pi2 * y) * (2 * cos(pi2 * x) - 1) + pis4 * sin(pi2 * x);
}

static SFEM_INLINE real_t rhs4_y(const real_t mu, const real_t x, const real_t y, const real_t z) {
    const real_t pis4 = 4 * M_PI * M_PI;
    const real_t pi2 = 2 * M_PI;
    return pis4 * mu * sin(pi2 * x) * (2 * cos(pi2 * y) - 1) - pis4 * sin(pi2 * y);
}

static SFEM_INLINE real_t rhs4_z(const real_t mu, const real_t x, const real_t y, const real_t z) {
    const real_t pis4 = 4 * M_PI * M_PI;
    const real_t pi2 = 2 * M_PI;
    return pis4 * mu * sin(pi2 * z) * (2 * cos(pi2 * x) - 1) - pis4 * sin(pi2 * y);
}

static SFEM_INLINE real_t rhs5_x(const real_t mu, const real_t x, const real_t y, const real_t z) {
    return 10000 * (x * x + y * y + z * z);
}

////////////////////////////////////////////////////////////////////////////////////

static void node_eval_f2D(const ptrdiff_t nnodes,
                          geom_t **const points,
                          const real_t mu,
                          real_t (*f)(const real_t, const real_t, const real_t),
                          real_t *values) {
    for (ptrdiff_t i = 0; i < nnodes; i++) {
        values[i] = f(mu, points[0][i], points[1][i]);
    }
}

static void node_eval_f3D(const ptrdiff_t nnodes,
                          geom_t **const points,
                          const real_t mu,
                          real_t (*f)(const real_t, const real_t, const real_t, const real_t),
                          real_t *values) {
    for (ptrdiff_t i = 0; i < nnodes; i++) {
        values[i] = f(mu, points[0][i], points[1][i], points[2][i]);
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
        case 4: {
            rhs_values[0] = calloc(mesh.nnodes, sizeof(real_t));
            rhs_values[1] = calloc(mesh.nnodes, sizeof(real_t));
            rhs_values[2] = calloc(mesh.nnodes, sizeof(real_t));
            node_eval_f3D(mesh.nnodes, mesh.points, SFEM_MU, &rhs4_x, rhs_values[0]);
            node_eval_f3D(mesh.nnodes, mesh.points, SFEM_MU, &rhs4_y, rhs_values[1]);
            node_eval_f3D(mesh.nnodes, mesh.points, SFEM_MU, &rhs4_z, rhs_values[2]);
            break;
        }
        case 5: {
            rhs_values[0] = calloc(mesh.nnodes, sizeof(real_t));
            node_eval_f3D(mesh.nnodes, mesh.points, SFEM_MU, &rhs5_x, rhs_values[0]);
            break;
        }
        default: {
            break;
        }
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

            if (0) {
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

        if (0) {
            // No static condensation contribution on RHS
            for (int i = 0; i < n_vars; i++) {
                if (rhs_values[i]) {
                    apply_mass(mesh.element_type,
                               mesh.nelements,
                               mesh.nnodes,
                               mesh.elements,
                               mesh.points,
                               rhs_values[i],
                               rhs[i]);
                }
            }
        } else {
            stokes_mini_assemble_rhs_soa(mesh.element_type,
                                         mesh.nelements,
                                         mesh.nnodes,
                                         mesh.elements,
                                         mesh.points,
                                         SFEM_MU,
                                         SFEM_RHO,
                                         rhs_values,
                                         rhs);
        }

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

            if (0) {
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
