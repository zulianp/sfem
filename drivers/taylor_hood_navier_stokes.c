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
#include "navier_stokes.h"
#include "read_mesh.h"

#include "isolver_lsolve.h"

//////////////////////////////////////////////

// TODOs 
// 1) Handle P2 - P1 mesh queries
// 2) Implement missing kernels for Tri6

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    MPI_Comm comm = MPI_COMM_WORLD;

    isolver_lsolve_t lsolve;
    lsolve.comm = comm;
    isolver_lsolve_init(&lsolve);

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
    real_t SFEM_DYNAMIC_VISCOSITY = 1;
    real_t SFEM_MASS_DENSITY = 1;
    int SFEM_PROBLEM_TYPE = 1;
    int SFEM_AOS = 0;

    SFEM_READ_ENV(SFEM_PROBLEM_TYPE, atoi);
    SFEM_READ_ENV(SFEM_DYNAMIC_VISCOSITY, atof);
    SFEM_READ_ENV(SFEM_MASS_DENSITY, atof);

    char *SFEM_VELOCITY_DIRICHLET_NODESET = 0;
    char *SFEM_VELOCITY_DIRICHLET_VALUE = 0;
    char *SFEM_VELOCITY_DIRICHLET_COMPONENT = 0;
    SFEM_READ_ENV(SFEM_VELOCITY_DIRICHLET_NODESET, );
    SFEM_READ_ENV(SFEM_VELOCITY_DIRICHLET_VALUE, );
    SFEM_READ_ENV(SFEM_VELOCITY_DIRICHLET_COMPONENT, );

    char * SFEM_PRESSURE_DIRICHLET_NODESET=0;

    char *SFEM_NEUMANN_SIDESET = 0;
    char *SFEM_NEUMANN_VALUE = 0;
    char *SFEM_NEUMANN_COMPONENT = 0;
    SFEM_READ_ENV(SFEM_NEUMANN_SIDESET, );
    SFEM_READ_ENV(SFEM_NEUMANN_VALUE, );
    SFEM_READ_ENV(SFEM_NEUMANN_COMPONENT, );

    int SFEM_MAX_IT = 1000;
    real_t SFEM_ATOL = 1e-8;
    real_t SFEM_RTOL = 1e-8;
    real_t SFEM_STOL = 1e-8;

    if (rank == 0) {
        printf(
            "----------------------------------------\n"
            "Options:\n"
            "----------------------------------------\n"
            "- SFEM_PROBLEM_TYPE=%d\n"
            "- SFEM_DYNAMIC_VISCOSITY=%g\n"
            "- SFEM_MASS_DENSITY=%g\n"
            "- SFEM_VELOCITY_DIRICHLET_NODESET=%s\n"
            "----------------------------------------\n",
            SFEM_PROBLEM_TYPE,
            SFEM_DYNAMIC_VISCOSITY,
            SFEM_MASS_DENSITY,
            SFEM_VELOCITY_DIRICHLET_NODESET);
    }

    // int n_neumann_conditions;
    // boundary_condition_t *neumann_conditions;

    int n_velocity_dirichlet_conditions;
    boundary_condition_t *velocity_dirichlet_conditions;

    int n_pressure_dirichlet_conditions;
    boundary_condition_t *pressure_dirichlet_conditions;

    read_dirichlet_conditions(&mesh,
                              SFEM_VELOCITY_DIRICHLET_NODESET,
                              SFEM_VELOCITY_DIRICHLET_VALUE,
                              SFEM_VELOCITY_DIRICHLET_COMPONENT,
                              &velocity_dirichlet_conditions,
                              &n_velocity_dirichlet_conditions);

    read_dirichlet_conditions(&mesh,
                              SFEM_PRESSURE_DIRICHLET_NODESET,
                              "",
                              "",
                              &velocity_dirichlet_conditions,
                              &n_velocity_dirichlet_conditions);

    // read_neumann_conditions(&mesh,
    //                         SFEM_NEUMANN_SIDESET,
    //                         SFEM_NEUMANN_VALUE,
    //                         SFEM_NEUMANN_COMPONENT,
    //                         &neumann_conditions,
    //                         &n_neumann_conditions);

    enum ElemType p1_type = elem_lower_order(mesh.element_type);
    const int sdim = elem_manifold_dim(mesh.element_type);
    const ptrdiff_t p1_nnodes = mesh.nnodes;  // TODO

    ptrdiff_t p1_nnz = 0;
    count_t *p1_rowptr = 0;
    idx_t *p1_colidx = 0;
    build_crs_graph_for_elem_type(
        p1_type, mesh.nelements, p1_nnodes, mesh.elements, &p1_rowptr, &p1_colidx);
    p1_nnz = p1_rowptr[mesh.nnodes];
    real_t *values = calloc(p1_nnz, sizeof(real_t));

    laplacian_assemble_hessian(p1_type,
                               mesh.nelements,
                               p1_nnodes,
                               mesh.elements,
                               mesh.points,
                               p1_rowptr,
                               p1_colidx,
                               values);

    for (int i = 0; i < n_pressure_dirichlet_conditions; i++) {
        crs_constraint_nodes_to_identity(pressure_dirichlet_conditions[i].local_size,
                                         pressure_dirichlet_conditions[i].idx,
                                         1,
                                         p1_rowptr,
                                         p1_colidx,
                                         values);
    }


    isolver_lsolve_set_max_iterations(&lsolve, SFEM_MAX_IT);
    isolver_lsolve_set_atol(&lsolve, SFEM_ATOL);
    isolver_lsolve_set_rtol(&lsolve, SFEM_RTOL);
    isolver_lsolve_set_stol(&lsolve, SFEM_STOL);
    isolver_lsolve_set_verbosity(&lsolve, 1);
    isolver_lsolve_update_crs(&lsolve, p1_nnodes, p1_nnodes, p1_rowptr, p1_colidx, values);

    real_t *vel[3];
    real_t *tentative_vel[3];
    real_t *buff = calloc(mesh.nnodes, sizeof(real_t));

    for (int d = 0; d < sdim; d++) {
        vel[d] = calloc(mesh.nnodes, sizeof(real_t));
        tentative_vel[d] = calloc(mesh.nnodes, sizeof(real_t));
    }

    real_t *p = calloc(p1_nnodes, sizeof(real_t));

    real_t T = 1;
    real_t dt = 0.1;
    for (real_t t = 0; t < T; t += dt) {
        //////////////////////////////////////////////////////////////
        // Tentative momentum step
        //////////////////////////////////////////////////////////////

        for (int d = 0; d < sdim; d++) {
            memset(tentative_vel[d], 0, mesh.nnodes * sizeof(real_t));
        }

        tri6_explict_momentum_tentative(mesh.nelements,
                                        mesh.nnodes,
                                        mesh.elements,
                                        mesh.points,
                                        dt,
                                        SFEM_DYNAMIC_VISCOSITY,
                                        vel,
                                        tentative_vel);

        for (int d = 0; d < sdim; d++) {
            // Write in place!!!
            apply_inv_lumped_mass(mesh.element_type,
                                  mesh.nelements,
                                  mesh.nnodes,
                                  mesh.elements,
                                  mesh.points,
                                  tentative_vel[d],
                                  tentative_vel[d]);
        }

        for (int i = 0; i < n_velocity_dirichlet_conditions; i++) {
            boundary_condition_t cond = velocity_dirichlet_conditions[i];
            constraint_nodes_to_value(
                cond.local_size, cond.idx, cond.value, tentative_vel[cond.component]);
        }

        //////////////////////////////////////////////////////////////
        // Poisson problem solve
        //////////////////////////////////////////////////////////////

        tri3_tri6_divergence(mesh.nelements,
                             mesh.nnodes,
                             mesh.elements,
                             mesh.points,
                             dt,
                             SFEM_DYNAMIC_VISCOSITY,
                             tentative_vel,
                             buff);

        memset(p, 0, p1_nnodes * sizeof(real_t));

        isolver_lsolve_t lsolve;
        lsolve.comm = comm;

        for (int i = 0; i < n_pressure_dirichlet_conditions; i++) {
            boundary_condition_t cond = pressure_dirichlet_conditions[i];
            assert(cond.component == 0);

            constraint_nodes_to_value(
                cond.local_size, cond.idx, cond.value, buff);
        }

        isolver_lsolve_apply(&lsolve, buff, p);

        //////////////////////////////////////////////////////////////
        // Correction/Projection step
        //////////////////////////////////////////////////////////////

        for (int d = 0; d < sdim; d++) {
            memset(vel[d], 0, mesh.nnodes * sizeof(real_t));
        }

        tri6_tri3_correction(
            mesh.nelements, mesh.nnodes, mesh.elements, mesh.points, dt, tentative_vel, p, vel);

        for (int d = 0; d < sdim; d++) {
            // Write in place!!!
            apply_inv_lumped_mass(mesh.element_type,
                                  mesh.nelements,
                                  mesh.nnodes,
                                  mesh.elements,
                                  mesh.points,
                                  vel[d],
                                  vel[d]);
        }

        for (int i = 0; i < n_velocity_dirichlet_conditions; i++) {
            boundary_condition_t cond = velocity_dirichlet_conditions[i];
            constraint_nodes_to_value(
                cond.local_size, cond.idx, cond.value, vel[cond.component]);
        }
    }

    char path[SFEM_MAX_PATH_LENGTH];
    for (int d = 0; d < sdim; d++) {
        sprintf(path, "%s/v.%d.raw", output_folder, d);
        array_write(comm, path, SFEM_MPI_REAL_T, vel[d], mesh.nnodes, mesh.nnodes);
    }

    sprintf(path, "%s/p.raw", output_folder);
    array_write(comm, path, SFEM_MPI_REAL_T, p, p1_nnodes, p1_nnodes);

    // Free resources

    free(p1_rowptr);
    free(p1_colidx);
    free(values);

    free(p);
    free(buff);

    for (int d = 0; d < mesh.spatial_dim; d++) {
        free(vel[d]);
        free(tentative_vel[d]);
    }

    ptrdiff_t nelements = mesh.nelements;
    ptrdiff_t nnodes = mesh.nnodes;

    mesh_destroy(&mesh);

    destroy_conditions(n_velocity_dirichlet_conditions, velocity_dirichlet_conditions);
    destroy_conditions(n_pressure_dirichlet_conditions, pressure_dirichlet_conditions);
    // destroy_conditions(n_neumann_conditions, neumann_conditions);

    double tock = MPI_Wtime();
    if (!rank) {
        printf("----------------------------------------\n");
        printf("#elements %ld #nodes %ld #nz %ld\n", (long)nelements, (long)nnodes, (long)p1_nnz);
        printf("TTS:\t\t\t%g seconds\n", tock - tick);
    }

    isolver_lsolve_destroy(&lsolve);
    return MPI_Finalize();
}
