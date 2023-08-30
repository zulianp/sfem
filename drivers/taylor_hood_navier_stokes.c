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

// https://fenicsproject.org/olddocs/dolfin/1.6.0/python/demo/documented/navier-stokes/python/documentation.html

idx_t max_idx(const ptrdiff_t n, const idx_t *idx) {
    idx_t ret = idx[0];

    for (ptrdiff_t i = 1; i < n; i++) {
        ret = MAX(ret, idx[i]);
    }

    return ret;
}

//////////////////////////////////////////////

#define N_SYSTEMS 3
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

    double tick = MPI_Wtime();

    ///////////////////////////////////////////////////////////////////////////////
    // Read data
    ///////////////////////////////////////////////////////////////////////////////

    const char *folder = argv[1];

    mesh_t mesh;
    if (mesh_read(comm, folder, &mesh)) {
        return EXIT_FAILURE;
    }

    real_t SFEM_DYNAMIC_VISCOSITY = 1;
    real_t SFEM_MASS_DENSITY = 1;
    SFEM_READ_ENV(SFEM_DYNAMIC_VISCOSITY, atof);
    SFEM_READ_ENV(SFEM_MASS_DENSITY, atof);

    char *SFEM_VELOCITY_DIRICHLET_NODESET = 0;
    char *SFEM_VELOCITY_DIRICHLET_VALUE = 0;
    char *SFEM_VELOCITY_DIRICHLET_COMPONENT = 0;
    SFEM_READ_ENV(SFEM_VELOCITY_DIRICHLET_NODESET, );
    SFEM_READ_ENV(SFEM_VELOCITY_DIRICHLET_VALUE, );
    SFEM_READ_ENV(SFEM_VELOCITY_DIRICHLET_COMPONENT, );

    char *SFEM_PRESSURE_DIRICHLET_NODESET = 0;
    SFEM_READ_ENV(SFEM_PRESSURE_DIRICHLET_NODESET, );

    char *SFEM_NEUMANN_SIDESET = 0;
    char *SFEM_NEUMANN_VALUE = 0;
    char *SFEM_NEUMANN_COMPONENT = 0;
    SFEM_READ_ENV(SFEM_NEUMANN_SIDESET, );
    SFEM_READ_ENV(SFEM_NEUMANN_VALUE, );
    SFEM_READ_ENV(SFEM_NEUMANN_COMPONENT, );

    int SFEM_MAX_IT = 1000;
    real_t SFEM_ATOL = 1e-10;
    real_t SFEM_RTOL = 1e-12;
    real_t SFEM_STOL = 1e-10;
    SFEM_READ_ENV(SFEM_MAX_IT, atoi);
    SFEM_READ_ENV(SFEM_ATOL, atof);
    SFEM_READ_ENV(SFEM_RTOL, atof);
    SFEM_READ_ENV(SFEM_STOL, atof);

    real_t SFEM_DT = 0.001;
    real_t SFEM_MAX_TIME = 1;
    SFEM_READ_ENV(SFEM_DT, atof);
    SFEM_READ_ENV(SFEM_MAX_TIME, atof);

    if (rank == 0) {
        printf(
            "----------------------------------------\n"
            "Options:\n"
            "----------------------------------------\n"
            "- SFEM_DYNAMIC_VISCOSITY=%g\n"
            "- SFEM_MASS_DENSITY=%g\n"
            "- SFEM_VELOCITY_DIRICHLET_NODESET=%s\n"
            "----------------------------------------\n",
            SFEM_DYNAMIC_VISCOSITY,
            SFEM_MASS_DENSITY,
            SFEM_VELOCITY_DIRICHLET_NODESET);
    }

    for (int s = 0; s < 2; s++) {
        isolver_lsolve_set_max_iterations(&lsolve[s], SFEM_MAX_IT);
        isolver_lsolve_set_atol(&lsolve[s], SFEM_ATOL);
        isolver_lsolve_set_rtol(&lsolve[s], SFEM_RTOL);
        isolver_lsolve_set_stol(&lsolve[s], SFEM_STOL);
        isolver_lsolve_set_verbosity(&lsolve[s], 1);
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
                              &pressure_dirichlet_conditions,
                              &n_pressure_dirichlet_conditions);

    // read_neumann_conditions(&mesh,
    //                         SFEM_NEUMANN_SIDESET,
    //                         SFEM_NEUMANN_VALUE,
    //                         SFEM_NEUMANN_COMPONENT,
    //                         &neumann_conditions,
    //                         &n_neumann_conditions);

    enum ElemType p1_type = elem_lower_order(mesh.element_type);
    const int p1_nxe = elem_num_nodes(p1_type);
    const int sdim = elem_manifold_dim(mesh.element_type);
    ptrdiff_t p1_nnodes = 0;

    for (int d = 0; d < p1_nxe; d++) {
        p1_nnodes = MAX(p1_nnodes, max_idx(mesh.nelements, mesh.elements[d]));
    }

    p1_nnodes += 1;

    ptrdiff_t p1_nnz = 0;
    count_t *p1_rowptr = 0;
    idx_t *p1_colidx = 0;
    build_crs_graph_for_elem_type(
        p1_type, mesh.nelements, p1_nnodes, mesh.elements, &p1_rowptr, &p1_colidx);
    p1_nnz = p1_rowptr[p1_nnodes];
    real_t *p1_values = calloc(p1_nnz, sizeof(real_t));

    laplacian_assemble_hessian(p1_type,
                               mesh.nelements,
                               p1_nnodes,
                               mesh.elements,
                               mesh.points,
                               p1_rowptr,
                               p1_colidx,
                               p1_values);

    for (int i = 0; i < n_pressure_dirichlet_conditions; i++) {
        crs_constraint_nodes_to_identity(pressure_dirichlet_conditions[i].local_size,
                                         pressure_dirichlet_conditions[i].idx,
                                         1,
                                         p1_rowptr,
                                         p1_colidx,
                                         p1_values);
    }

    isolver_lsolve_update_crs(&lsolve[0], p1_nnodes, p1_nnodes, p1_rowptr, p1_colidx, p1_values);

    ptrdiff_t p2_nnz = 0;
    count_t *p2_rowptr = 0;
    idx_t *p2_colidx = 0;
    build_crs_graph_for_elem_type(
        mesh.element_type, mesh.nelements, mesh.nnodes, mesh.elements, &p2_rowptr, &p2_colidx);
    p2_nnz = p2_rowptr[mesh.nnodes];
    real_t *p2_momentum = calloc(p2_nnz, sizeof(real_t));
    real_t *p2_projection = calloc(p2_nnz, sizeof(real_t));

    {
        // Tentative Momentum Step
        // FIXME compute actual LHS
        // assemble_mass(mesh.element_type,
        //               mesh.nelements,
        //               mesh.nnodes,
        //               mesh.elements,
        //               mesh.points,
        //               p2_rowptr,
        //               p2_colidx,
        //               p2_momentum);

        // for (int i = 0; i < n_velocity_dirichlet_conditions; i++) {
        //     crs_constraint_nodes_to_identity(velocity_dirichlet_conditions[i].local_size,
        //                                      velocity_dirichlet_conditions[i].idx,
        //                                      1,
        //                                      p2_rowptr,
        //                                      p2_colidx,
        //                                      p2_momentum);
        // }

        // isolver_lsolve_update_crs(
        //     &lsolve[1], mesh.nnodes, mesh.nnodes, p2_rowptr, p2_colidx, p2_momentum);
    }

    {
        // Projection Step
        // FIXME compute actual LHS
        // assemble_mass(mesh.element_type,
        //               mesh.nelements,
        //               mesh.nnodes,
        //               mesh.elements,
        //               mesh.points,
        //               p2_rowptr,
        //               p2_colidx,
        //               p2_momentum);

        // for (int i = 0; i < n_velocity_dirichlet_conditions; i++) {
        //     crs_constraint_nodes_to_identity(velocity_dirichlet_conditions[i].local_size,
        //                                      velocity_dirichlet_conditions[i].idx,
        //                                      1,
        //                                      p2_rowptr,
        //                                      p2_colidx,
        //                                      p2_momentum);
        // }

        // isolver_lsolve_update_crs(
        //     &lsolve[1], mesh.nnodes, mesh.nnodes, p2_rowptr, p2_colidx, p2_momentum);
    }

    real_t *vel[3];
    real_t *correction[3];
    real_t *tentative_vel[3];
    real_t *buff = calloc(p1_nnodes, sizeof(real_t));

    for (int d = 0; d < sdim; d++) {
        vel[d] = calloc(mesh.nnodes, sizeof(real_t));
        tentative_vel[d] = calloc(mesh.nnodes, sizeof(real_t));
        correction[d] = calloc(mesh.nnodes, sizeof(real_t));
    }

    real_t *p = calloc(p1_nnodes, sizeof(real_t));

    for (real_t t = 0; t < SFEM_MAX_TIME; t += SFEM_DT) {
        //////////////////////////////////////////////////////////////
        // Tentative momentum step
        //////////////////////////////////////////////////////////////
        {
            for (int d = 0; d < sdim; d++) {
                memset(tentative_vel[d], 0, mesh.nnodes * sizeof(real_t));
            }

            // FIXME compute actual RHS
            tri6_explict_momentum_tentative(mesh.nelements,
                                            mesh.nnodes,
                                            mesh.elements,
                                            mesh.points,
                                            SFEM_DT,
                                            SFEM_DYNAMIC_VISCOSITY,
                                            vel,
                                            tentative_vel);

            for (int i = 0; i < n_velocity_dirichlet_conditions; i++) {
                boundary_condition_t cond = velocity_dirichlet_conditions[i];
                constraint_nodes_to_value(
                    cond.local_size, cond.idx, cond.value, tentative_vel[cond.component]);
            }

            for (int d = 0; d < sdim; d++) {
                memset(correction[d], 0, mesh.nnodes * sizeof(real_t));
                isolver_lsolve_apply(&lsolve[1], tentative_vel[d], correction[d]);
                memcpy(tentative_vel[d], correction[d], mesh.nnodes * sizeof(real_t));
            }

            for (int i = 0; i < n_velocity_dirichlet_conditions; i++) {
                boundary_condition_t cond = velocity_dirichlet_conditions[i];
                constraint_nodes_to_value(
                    cond.local_size, cond.idx, cond.value, tentative_vel[cond.component]);
            }
        }
        //////////////////////////////////////////////////////////////
        // Poisson problem + solve
        //////////////////////////////////////////////////////////////
        {
            tri3_tri6_divergence(mesh.nelements,
                                 mesh.nnodes,
                                 mesh.elements,
                                 mesh.points,
                                 SFEM_DT,
                                 SFEM_MASS_DENSITY,
                                 SFEM_DYNAMIC_VISCOSITY,
                                 tentative_vel,
                                 buff);

            memset(p, 0, p1_nnodes * sizeof(real_t));

            for (int i = 0; i < n_pressure_dirichlet_conditions; i++) {
                boundary_condition_t cond = pressure_dirichlet_conditions[i];
                assert(cond.component == 0);

                constraint_nodes_to_value(cond.local_size, cond.idx, cond.value, buff);
            }

            isolver_lsolve_apply(&lsolve[0], buff, p);
        }
        //////////////////////////////////////////////////////////////
        // Correction/Projection step
        //////////////////////////////////////////////////////////////
        {
            // for (int d = 0; d < sdim; d++) {
            //     memset(vel[d], 0, mesh.nnodes * sizeof(real_t));
            // }

            tri6_tri3_correction(mesh.nelements,
                                 mesh.nnodes,
                                 mesh.elements,
                                 mesh.points,
                                 SFEM_DT,
                                 SFEM_MASS_DENSITY,
                                 tentative_vel,
                                 p,
                                 correction);

            for (int i = 0; i < n_velocity_dirichlet_conditions; i++) {
                boundary_condition_t cond = velocity_dirichlet_conditions[i];
                constraint_nodes_to_value(
                    cond.local_size, cond.idx, cond.value, correction[cond.component]);
            }

            for (int d = 0; d < sdim; d++) {
                memset(tentative_vel[d], 0, mesh.nnodes * sizeof(real_t));
                isolver_lsolve_apply(&lsolve[1], correction[d], tentative_vel[d]);
            }

            for (int d = 0; d < sdim; d++) {
                for (ptrdiff_t i = 0; i < mesh.nnodes; i++) {
                    vel[d][i] += tentative_vel[d][i];
                }
            }

            for (int i = 0; i < n_velocity_dirichlet_conditions; i++) {
                boundary_condition_t cond = velocity_dirichlet_conditions[i];
                constraint_nodes_to_value(
                    cond.local_size, cond.idx, cond.value, vel[cond.component]);
            }
        }
    }

    char path[SFEM_MAX_PATH_LENGTH];
    for (int d = 0; d < sdim; d++) {
        sprintf(path, "%s/v.%d.raw", output_folder, d);
        array_write(comm, path, SFEM_MPI_REAL_T, vel[d], mesh.nnodes, mesh.nnodes);
    }

    for (int d = 0; d < sdim; d++) {
        sprintf(path, "%s/c.%d.raw", output_folder, d);
        array_write(comm, path, SFEM_MPI_REAL_T, tentative_vel[d], mesh.nnodes, mesh.nnodes);
    }

    // for (int d = 0; d < sdim; d++) {
    //     sprintf(path, "%s/c.%d.raw", output_folder, d);
    //     array_write(comm, path, SFEM_MPI_REAL_T, correction[d], mesh.nnodes, mesh.nnodes);
    // }

    sprintf(path, "%s/p.raw", output_folder);
    array_write(comm, path, SFEM_MPI_REAL_T, p, p1_nnodes, p1_nnodes);

    sprintf(path, "%s/tp.raw", output_folder);
    array_write(comm, path, SFEM_MPI_REAL_T, buff, p1_nnodes, p1_nnodes);

    // Free resources
    free(p1_rowptr);
    free(p1_colidx);
    free(p1_values);

    free(p2_rowptr);
    free(p2_colidx);
    free(p2_momentum);
    free(p2_projection);

    free(p);
    free(buff);

    for (int d = 0; d < sdim; d++) {
        free(vel[d]);
        free(tentative_vel[d]);
        free(correction[d]);
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

    // FIXME One is enough for now, but it is not clean
    isolver_lsolve_destroy(&lsolve[0]);
    // isolver_lsolve_destroy(&lsolve[1]);
    return MPI_Finalize();
}
