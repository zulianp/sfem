#include <math.h>
#include <stddef.h>
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

#include "boundary_condition.h"
#include "boundary_condition_io.h"
#include "dirichlet.h"
#include "neumann.h"

#include "laplacian.h"
#include "navier_stokes.h"
#include "read_mesh.h"
#include "cvfem_operators.h"

#include "sfem_logger.h"

void axpy(const ptrdiff_t n, const real_t alpha, const real_t *const x, real_t *const y) {
#pragma omp parallel for
    for (ptrdiff_t i = 0; i < n; i++) {
        y[i] += alpha * x[i];
    }
}

void scal(const ptrdiff_t n, const real_t alpha, real_t *const x) {
#pragma omp parallel for
    for (ptrdiff_t i = 0; i < n; i++) {
        x[i] *= alpha;
    }
}

real_t dot(const ptrdiff_t n, const real_t *const x, const real_t *const y) {
    real_t ret = 0;
#pragma omp parallel for reduction(+ : ret)
    for (ptrdiff_t i = 0; i < n; i++) {
        ret += y[i] * x[i];
    }

    return ret;
}

void ediv(const ptrdiff_t n,
          const real_t *const nom,
          const real_t *const denom,
          real_t *const result) {
#pragma omp parallel for
    for (ptrdiff_t i = 0; i < n; i++) {
        result[i] = nom[i] / denom[i];
    }
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
        fprintf(stderr, "usage: %s <folder> <output>\n", argv[0]);
        return EXIT_FAILURE;
    }

    const char *output_folder = argv[2];

    struct stat st = {0};
    if (stat(output_folder, &st) == -1) {
        mkdir(output_folder, 0700);
    }

    char path[SFEM_MAX_PATH_LENGTH];
    double tick = MPI_Wtime();

    ///////////////////////////////////////////////////////////////////////////////
    // Read data
    ///////////////////////////////////////////////////////////////////////////////

    const char *folder = argv[1];

    mesh_t mesh;
    if (mesh_read(comm, folder, &mesh)) {
        return EXIT_FAILURE;
    }
    char *SFEM_DIRICHLET_NODESET = 0;
    char *SFEM_DIRICHLET_VALUE = 0;
    SFEM_READ_ENV(SFEM_DIRICHLET_NODESET, );
    SFEM_READ_ENV(SFEM_DIRICHLET_VALUE, );

    char *SFEM_NEUMANN_SIDESET = 0;
    char *SFEM_NEUMANN_VALUE = 0;
    SFEM_READ_ENV(SFEM_NEUMANN_SIDESET, );
    SFEM_READ_ENV(SFEM_NEUMANN_VALUE, );

    float SFEM_DT = 0.1;
    float SFEM_MAX_TIME = 1;
    float SFEM_EXPORT_FREQUENCY = 1;
    float SFEM_DIFFUSIVITY = 1;

    SFEM_READ_ENV(SFEM_DT, atof);
    SFEM_READ_ENV(SFEM_MAX_TIME, atof);
    SFEM_READ_ENV(SFEM_EXPORT_FREQUENCY, atof);
    SFEM_READ_ENV(SFEM_DIFFUSIVITY, atof);

    const char *SFEM_RESTART_FOLDER = 0;
    SFEM_READ_ENV(SFEM_RESTART_FOLDER, );

    int SFEM_RESTART_ID = 0;
    SFEM_READ_ENV(SFEM_RESTART_ID, atoi);

    if (rank == 0) {
        printf(
            "----------------------------------------\n"
            "Options:\n"
            "----------------------------------------\n"
            "- SFEM_DT=%g\n"
            "- SFEM_RESTART_FOLDER=%s\n"
            "- SFEM_RESTART_ID=%d\n"
            "----------------------------------------\n",
            SFEM_DT,
            SFEM_RESTART_FOLDER,
            SFEM_RESTART_ID);
    }

    if (SFEM_RESTART_FOLDER && !SFEM_RESTART_ID) {
        fprintf(stderr, "Defined SFEM_RESTART_FOLDER but not SFEM_RESTART_ID\n");
        return EXIT_FAILURE;
    }

    int n_dirichlet_conditions;
    boundary_condition_t *dirichlet_conditions;
    read_dirichlet_conditions(&mesh,
                              SFEM_DIRICHLET_NODESET,
                              SFEM_DIRICHLET_VALUE,
                              "0",
                              &dirichlet_conditions,
                              &n_dirichlet_conditions);

    const int sdim = elem_manifold_dim(mesh.element_type);

    real_t *buff = calloc(mesh.nnodes, sizeof(real_t));
    real_t *c = calloc(mesh.nnodes, sizeof(real_t));

    real_t *vel[3];
    real_t *update;

    for (int d = 0; d < sdim; d++) {
        vel[d] = calloc(mesh.nnodes, sizeof(real_t));
    }

    for (ptrdiff_t i = 0; i < mesh.nnodes; i++) {
        vel[0][i] = 1;
        // vel[1][i] = 0.01;
    }

    update = calloc(mesh.nnodes, sizeof(real_t));

    // FIXME these are initial conditions (rename)
    for (int i = 0; i < n_dirichlet_conditions; i++) {
        boundary_condition_t cond = dirichlet_conditions[i];
        constraint_nodes_to_value(cond.local_size, cond.idx, cond.value, c);
    }

    int export_counter = 0;
    real_t next_check_point = SFEM_EXPORT_FREQUENCY;
    logger_t time_logger;

    if (SFEM_RESTART_FOLDER) {
        sprintf(path, "%s/c.%09d.raw", SFEM_RESTART_FOLDER, SFEM_RESTART_ID);
        if (array_read(comm, path, SFEM_MPI_REAL_T, (void *)c, mesh.nnodes, mesh.nnodes)) {
            fprintf(stderr, "Error reading restart file: %s\n", SFEM_RESTART_FOLDER);
            return EXIT_FAILURE;
        }

        export_counter = SFEM_RESTART_ID + 1;
    } else {  // Write to disk
        printf("%g/%g\n", 0., SFEM_MAX_TIME);
        sprintf(path, "%s/c.%09d.raw", output_folder, export_counter);
        array_write(comm, path, SFEM_MPI_REAL_T, c, mesh.nnodes, mesh.nnodes);
        sprintf(path, "%s/time.txt", output_folder);
        log_create_file(&time_logger, path, "w");
        log_write_double(&time_logger, 0);
        export_counter++;
    }

    real_t *cv_volumes = calloc(mesh.nnodes, sizeof(real_t));
    cvfem_cv_volumes(
        mesh.element_type, mesh.nelements, mesh.nnodes, mesh.elements, mesh.points, cv_volumes);

    real_t dt = SFEM_DT;
    ptrdiff_t step_count = 0;
    for (real_t t = 0; t < SFEM_MAX_TIME; t += dt, step_count++) {
        // Integrate
        memset(buff, 0, mesh.nnodes * sizeof(real_t));

        if (SFEM_DIFFUSIVITY != 0.) {
            cvfem_laplacian_apply(mesh.element_type,
                                  mesh.nelements,
                                  mesh.nnodes,
                                  mesh.elements,
                                  mesh.points,
                                  c,
                                  buff);
            scal(mesh.nnodes, -SFEM_DIFFUSIVITY, buff);
        }

        cvfem_convection_apply(mesh.element_type,
                               mesh.nelements,
                               mesh.nnodes,
                               mesh.elements,
                               mesh.points,
                               vel,
                               c,
                               buff);

        ediv(mesh.nnodes, buff, cv_volumes, update);
        axpy(mesh.nnodes, SFEM_DT, update, c);

        real_t integr_concentration = dot(mesh.nnodes, c, cv_volumes);

        // for (int i = 0; i < n_dirichlet_conditions; i++) {
        //     boundary_condition_t cond = dirichlet_conditions[i];
        //     constraint_nodes_to_value(cond.local_size, cond.idx, cond.value, c);
        // }

        if (t >= (next_check_point - SFEM_DT / 2)) {  // Write to disk
            printf("%g/%g dt=%g mc=%g\n", t, SFEM_MAX_TIME, dt, integr_concentration);

            sprintf(path, "%s/c.%09d.raw", output_folder, export_counter);
            array_write(comm, path, SFEM_MPI_REAL_T, c, mesh.nnodes, mesh.nnodes);

            log_write_double(&time_logger, t);

            next_check_point += SFEM_EXPORT_FREQUENCY;
            export_counter++;
        }
    }

    sprintf(path, "%s/cv_volumes.raw", output_folder);
    array_write(comm, path, SFEM_MPI_REAL_T, cv_volumes, mesh.nnodes, mesh.nnodes);

    ptrdiff_t nelements = mesh.nelements;
    ptrdiff_t nnodes = mesh.nnodes;

    mesh_destroy(&mesh);
    destroy_conditions(n_dirichlet_conditions, dirichlet_conditions);
    // destroy_conditions(n_neumann_conditions, neumann_conditions);

    log_destroy(&time_logger);

    free(update);
    free(buff);
    free(c);
    free(cv_volumes);

    for (int d = 0; d < sdim; d++) {
        free(vel[d]);
    }

    double tock = MPI_Wtime();
    if (!rank) {
        printf("----------------------------------------\n");
        printf("#elements %ld #nodes %ld #steps %ld\n", (long)nelements, (long)nnodes, step_count);
        printf("TTS:\t\t\t%g seconds\n", tock - tick);
    }

    return MPI_Finalize();
}
