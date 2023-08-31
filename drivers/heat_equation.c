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

#include "boundary_condition.h"
#include "dirichlet.h"
#include "neumann.h"

#include "laplacian.h"
#include "read_mesh.h"

#include "isolver_lsolve.h"

#include "spmv.h"

//////////////////////////////////////////////

#define N_SYSTEMS 2
#define INVERSE_SYSTEM 0
#define INVERSE_MASS_MATRIX 1

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    MPI_Comm comm = MPI_COMM_WORLD;
    isolver_lsolve_t lsolve[N_SYSTEMS];

    for (int s = 0; s < N_SYSTEMS; s++) {
        lsolve[s].comm = comm;
        isolver_lsolve_init(&lsolve[s]);
    }

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

    real_t SFEM_DIFFUSIVITY = 1;
    SFEM_READ_ENV(SFEM_DIFFUSIVITY, atof);

    char *SFEM_DIRICHLET_NODESET = 0;
    char *SFEM_DIRICHLET_VALUE = 0;
    char *SFEM_DIRICHLET_COMPONENT = 0;
    SFEM_READ_ENV(SFEM_DIRICHLET_NODESET, );
    SFEM_READ_ENV(SFEM_DIRICHLET_VALUE, );
    SFEM_READ_ENV(SFEM_DIRICHLET_COMPONENT, );

    char *SFEM_NEUMANN_SIDESET = 0;
    char *SFEM_NEUMANN_VALUE = 0;
    char *SFEM_NEUMANN_COMPONENT = 0;
    SFEM_READ_ENV(SFEM_NEUMANN_SIDESET, );
    SFEM_READ_ENV(SFEM_NEUMANN_VALUE, );
    SFEM_READ_ENV(SFEM_NEUMANN_COMPONENT, );

    int SFEM_MAX_IT = 1000;
    real_t SFEM_ATOL = 1e-15;
    real_t SFEM_RTOL = 1e-14;
    real_t SFEM_STOL = 1e-12;
    SFEM_READ_ENV(SFEM_MAX_IT, atoi);
    SFEM_READ_ENV(SFEM_ATOL, atof);
    SFEM_READ_ENV(SFEM_RTOL, atof);
    SFEM_READ_ENV(SFEM_STOL, atof);

    real_t SFEM_DT = 1;
    real_t SFEM_MAX_TIME = 1;
    SFEM_READ_ENV(SFEM_DT, atof);
    SFEM_READ_ENV(SFEM_MAX_TIME, atof);

    int SFEM_IMPLICIT = 1;
    SFEM_READ_ENV(SFEM_IMPLICIT, atoi);

    if (rank == 0) {
        printf(
            "----------------------------------------\n"
            "Options:\n"
            "----------------------------------------\n"
            "- SFEM_DT=%g\n"
            "- SFEM_DIFFUSIVITY=%g\n"
            "- SFEM_DIRICHLET_NODESET=%s\n"
            "- SFEM_IMPLICIT=%d\n"
            "----------------------------------------\n",
            SFEM_DT,
            SFEM_DIFFUSIVITY,
            SFEM_DIRICHLET_NODESET,
            SFEM_IMPLICIT);
    }

    for (int s = 0; s < N_SYSTEMS; s++) {
        isolver_lsolve_set_max_iterations(&lsolve[s], SFEM_MAX_IT);
        isolver_lsolve_set_atol(&lsolve[s], SFEM_ATOL);
        isolver_lsolve_set_rtol(&lsolve[s], SFEM_RTOL);
        isolver_lsolve_set_stol(&lsolve[s], SFEM_STOL);
        isolver_lsolve_set_verbosity(&lsolve[s], 1);
    }

    int n_dirichlet_conditions;
    boundary_condition_t *dirichlet_conditions;

    read_dirichlet_conditions(&mesh,
                              SFEM_DIRICHLET_NODESET,
                              SFEM_DIRICHLET_VALUE,
                              SFEM_DIRICHLET_COMPONENT,
                              &dirichlet_conditions,
                              &n_dirichlet_conditions);

    // read_neumann_conditions(&mesh,
    //                         SFEM_NEUMANN_SIDESET,
    //                         SFEM_NEUMANN_VALUE,
    //                         SFEM_NEUMANN_COMPONENT,
    //                         &neumann_conditions,
    //                         &n_neumann_conditions);

    const int sdim = elem_manifold_dim(mesh.element_type);

    count_t *rowptr = 0;
    idx_t *colidx = 0;
    real_t *diffusion = 0;
    real_t *mass_matrix = 0;
    real_t *system_matrix = 0;

    build_crs_graph_for_elem_type(
        mesh.element_type, mesh.nelements, mesh.nnodes, mesh.elements, &rowptr, &colidx);

    ptrdiff_t nnz = rowptr[mesh.nnodes];
    mass_matrix = calloc(nnz, sizeof(real_t));
    system_matrix = calloc(nnz, sizeof(real_t));

    laplacian_assemble_hessian(mesh.element_type,
                               mesh.nelements,
                               mesh.nnodes,
                               mesh.elements,
                               mesh.points,
                               rowptr,
                               colidx,
                               system_matrix);

    assemble_mass(mesh.element_type,
                  mesh.nelements,
                  mesh.nnodes,
                  mesh.elements,
                  mesh.points,
                  rowptr,
                  colidx,
                  mass_matrix);

    for (ptrdiff_t i = 0; i < nnz; i++) {
        system_matrix[i] *= SFEM_DT;

        if(SFEM_IMPLICIT) {
            system_matrix[i] += mass_matrix[i];
        } else {
            system_matrix[i] = mass_matrix[i] - system_matrix[i];
        }
    }

    for (int i = 0; i < n_dirichlet_conditions; i++) {
        crs_constraint_nodes_to_identity(dirichlet_conditions[i].local_size,
                                         dirichlet_conditions[i].idx,
                                         1,
                                         rowptr,
                                         colidx,
                                         mass_matrix);
    }

    for (int i = 0; i < n_dirichlet_conditions; i++) {
        crs_constraint_nodes_to_identity(dirichlet_conditions[i].local_size,
                                         dirichlet_conditions[i].idx,
                                         1,
                                         rowptr,
                                         colidx,
                                         system_matrix);
    }

    isolver_lsolve_update_crs(
        &lsolve[INVERSE_SYSTEM], mesh.nnodes, mesh.nnodes, rowptr, colidx, system_matrix);

    isolver_lsolve_update_crs(
        &lsolve[INVERSE_MASS_MATRIX], mesh.nnodes, mesh.nnodes, rowptr, colidx, mass_matrix);

    real_t *u = calloc(mesh.nnodes, sizeof(real_t));
    real_t *u_old = calloc(mesh.nnodes, sizeof(real_t));
    
    if (SFEM_IMPLICIT) {
        int step_count = 0;
        for (real_t t = 0; t < SFEM_MAX_TIME; t += SFEM_DT, step_count++) {
            spmv_crs(mesh.nnodes, rowptr, colidx, mass_matrix, u, u_old);

            for (int i = 0; i < n_dirichlet_conditions; i++) {
                boundary_condition_t cond = dirichlet_conditions[i];
                constraint_nodes_to_value(cond.local_size, cond.idx, cond.value, u_old);
            }

            isolver_lsolve_apply(&lsolve[INVERSE_SYSTEM], u_old, u);

            printf("%g/%g\n", t, SFEM_MAX_TIME);
            sprintf(path, "%s/u.%05d.raw", output_folder, step_count);
            array_write(comm, path, SFEM_MPI_REAL_T, u, mesh.nnodes, mesh.nnodes);
        }

    } else {
        int step_count = 0;
        for (real_t t = 0; t < SFEM_MAX_TIME; t += SFEM_DT, step_count++) {
            const int i = step_count % 2;
            const int ip1 = (step_count + 1) % 2;

            spmv_crs(mesh.nnodes, rowptr, colidx, system_matrix, u_old, u);

            for (int i = 0; i < n_dirichlet_conditions; i++) {
                boundary_condition_t cond = dirichlet_conditions[i];
                constraint_nodes_to_value(cond.local_size, cond.idx, cond.value, u);
            }

            // isolver_lsolve_apply(&lsolve[INVERSE_MASS_MATRIX], u, u_old);
            apply_inv_lumped_mass(mesh.element_type,
                                  mesh.nelements,
                                  mesh.nnodes,
                                  mesh.elements,
                                  mesh.points,
                                  u,
                                  u_old);

            printf("%g/%g\n", t, SFEM_MAX_TIME);
            sprintf(path, "%s/u.%05d.raw", output_folder, step_count);
            array_write(comm, path, SFEM_MPI_REAL_T, u_old, mesh.nnodes, mesh.nnodes);
        }
    }

    free(rowptr);
    free(colidx);
    free(diffusion);
    free(mass_matrix);
    free(system_matrix);
    free(u);
    free(u_old);

    ptrdiff_t nelements = mesh.nelements;
    ptrdiff_t nnodes = mesh.nnodes;

    mesh_destroy(&mesh);
    destroy_conditions(n_dirichlet_conditions, dirichlet_conditions);
    // destroy_conditions(n_neumann_conditions, neumann_conditions);

    double tock = MPI_Wtime();
    if (!rank) {
        printf("----------------------------------------\n");
        printf("#elements %ld #nodes %ld\n", (long)nelements, (long)nnodes);
        printf("TTS:\t\t\t%g seconds\n", tock - tick);
    }

    // FIXME One is enough for now, but it is not clean
    isolver_lsolve_destroy(&lsolve[0]);
    // isolver_lsolve_destroy(&lsolve[1]);
    return MPI_Finalize();
}
