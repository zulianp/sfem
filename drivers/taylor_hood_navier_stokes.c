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

#include "boundary_condition.h"
#include "boundary_condition_io.h"
#include "dirichlet.h"
#include "neumann.h"

#include "laplacian.h"
#include "navier_stokes.h"
#include "read_mesh.h"

#include "isolver_lsolve.h"

#include "constrained_gs.h"

#include "sfem_logger.h"

// https://fenicsproject.org/olddocs/dolfin/1.6.0/python/demo/documented/navier-stokes/python/documentation.html

// Explicit euler
// #elements 65536 #nodes 131585 #nz 0 #steps 10001
// 583.669 / 10001 = 0.0583669 (seconds) per time-step

ptrdiff_t remove_p2_nodes(const ptrdiff_t n, const idx_t bound_p1, idx_t *idx) {
    ptrdiff_t nret = sortreduce(idx, n);

    for (ptrdiff_t i = 0; i < n; i++) {
        if (idx[i] >= bound_p1) {
            return i;
        }
    }

    return nret;
}

idx_t max_idx(const ptrdiff_t n, const idx_t *idx) {
    idx_t ret = idx[0];

    for (ptrdiff_t i = 1; i < n; i++) {
        ret = MAX(ret, idx[i]);
    }

    return ret;
}

static ptrdiff_t count_nan(const ptrdiff_t n, const real_t *const v) {
    ptrdiff_t ret = 0;
    for (ptrdiff_t i = 0; i < n; i++) {
        ret += !(v[i] == v[i]);
    }
    return ret;
}

static void make_matrix_nonsingular(const real_t factor,
                                    const ptrdiff_t n,
                                    const count_t *const SFEM_RESTRICT rowptr,
                                    const idx_t *const SFEM_RESTRICT colidx,
                                    real_t *const SFEM_RESTRICT values) {
    count_t rstart = rowptr[n - 1];
    count_t rextent = rowptr[n] - rstart;
    const idx_t *const cols = &colidx[rstart];
    real_t *const vals = &values[rstart];

    for (count_t i = 0; i < rextent; ++i) {
        if (cols[i] == n - 1) {
            vals[i] *= (1 + factor);

            // vals[i] += factor;
            break;
        }
    }
}

static void shift_diag(const ptrdiff_t n,
                       const real_t factor,
                       const real_t *const SFEM_RESTRICT vector,
                       const count_t *const SFEM_RESTRICT rowptr,
                       const idx_t *const SFEM_RESTRICT colidx,
                       real_t *const SFEM_RESTRICT values) {
    for (ptrdiff_t i = 0; i < n; ++i) {
        // ptrdiff_t i = n - 1;
        count_t rstart = rowptr[i];
        count_t rextent = rowptr[i + 1] - rstart;
        const idx_t *const cols = &colidx[rstart];
        real_t *const vals = &values[rstart];

        for (count_t k = 0; k < rextent; ++k) {
            if (cols[k] == i) {
                vals[k] += factor * vector[i];
                break;
            }
        }
    }
}

//////////////////////////////////////////////

#define N_SYSTEMS 3
#define INVERSE_MASS_MATRIX 2
#define INVERSE_POISSON_MATRIX 0

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    if (sizeof(count_t) != sizeof(isolver_idx_t)) {
        fprintf(stderr,
                "%s Incompatible types for isolver (count_t != isolver_idx_t)\n",
                argv[0]);
        return EXIT_FAILURE;
    }

    if (sizeof(real_t) != sizeof(isolver_scalar_t)) {
        fprintf(stderr,
                "%s Incompatible types for isolver (real_t != isolver_scalar_t)\n",
                argv[0]);
        return EXIT_FAILURE;
    }

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

    real_t SFEM_VISCOSITY = 1;
    real_t SFEM_MASS_DENSITY = 1;
    SFEM_READ_ENV(SFEM_VISCOSITY, atof);
    SFEM_READ_ENV(SFEM_MASS_DENSITY, atof);

    char *SFEM_VELOCITY_DIRICHLET_NODESET = 0;
    char *SFEM_VELOCITY_DIRICHLET_VALUE = 0;
    char *SFEM_VELOCITY_DIRICHLET_COMPONENT = 0;
    SFEM_READ_ENV(SFEM_VELOCITY_DIRICHLET_NODESET, );
    SFEM_READ_ENV(SFEM_VELOCITY_DIRICHLET_VALUE, );
    SFEM_READ_ENV(SFEM_VELOCITY_DIRICHLET_COMPONENT, );

    char *SFEM_PRESSURE_DIRICHLET_NODESET = 0;
    char *SFEM_PRESSURE_DIRICHLET_VALUE = 0;
    char *SFEM_PRESSURE_DIRICHLET_COMPONENT = 0;
    SFEM_READ_ENV(SFEM_PRESSURE_DIRICHLET_NODESET, );
    SFEM_READ_ENV(SFEM_PRESSURE_DIRICHLET_VALUE, );
    SFEM_READ_ENV(SFEM_PRESSURE_DIRICHLET_COMPONENT, );

    char *SFEM_NEUMANN_SIDESET = 0;
    char *SFEM_NEUMANN_VALUE = 0;
    char *SFEM_NEUMANN_COMPONENT = 0;
    SFEM_READ_ENV(SFEM_NEUMANN_SIDESET, );
    SFEM_READ_ENV(SFEM_NEUMANN_VALUE, );
    SFEM_READ_ENV(SFEM_NEUMANN_COMPONENT, );

    int SFEM_MAX_IT = 100;
    real_t SFEM_ATOL = 1e-15;
    real_t SFEM_RTOL = 1e-14;
    real_t SFEM_STOL = 1e-12;
    int SFEM_VERBOSE = 0;

    SFEM_READ_ENV(SFEM_MAX_IT, atoi);
    SFEM_READ_ENV(SFEM_ATOL, atof);
    SFEM_READ_ENV(SFEM_RTOL, atof);
    SFEM_READ_ENV(SFEM_STOL, atof);
    SFEM_READ_ENV(SFEM_VERBOSE, atoi);

    real_t SFEM_DT = 1;
    real_t SFEM_MAX_TIME = 1;
    real_t SFEM_CFL = 0.1;
    real_t SFEM_EXPORT_FREQUENCY = 0.1;
    int SFEM_LUMPED_MASS = 0;

    SFEM_READ_ENV(SFEM_DT, atof);
    SFEM_READ_ENV(SFEM_MAX_TIME, atof);
    SFEM_READ_ENV(SFEM_CFL, atof);
    SFEM_READ_ENV(SFEM_EXPORT_FREQUENCY, atof);
    SFEM_READ_ENV(SFEM_LUMPED_MASS, atoi);

    // int SFEM_AVG_PRESSURE_CONSTRAINT = 1;
    int SFEM_AVG_PRESSURE_CONSTRAINT = 0;
    SFEM_READ_ENV(SFEM_AVG_PRESSURE_CONSTRAINT, atoi);

    real_t SFEM_REGULARIZATION_FACTOR = 1e-8;
    SFEM_READ_ENV(SFEM_REGULARIZATION_FACTOR, atof);

    const char *SFEM_RESTART_FOLDER = 0;
    SFEM_READ_ENV(SFEM_RESTART_FOLDER, );

    int SFEM_RESTART_ID = 0;
    SFEM_READ_ENV(SFEM_RESTART_ID, atoi);

    if (rank == 0) {
        printf("----------------------------------------\n"
               "Options:\n"
               "----------------------------------------\n"
               "- SFEM_DT=%g\n"
               "- SFEM_CFL=%g\n"
               "- SFEM_LUMPED_MASS=%d\n"
               "- SFEM_VISCOSITY=%g\n"
               "- SFEM_MASS_DENSITY=%g\n"
               "- SFEM_VELOCITY_DIRICHLET_NODESET=%s\n"
               "- SFEM_AVG_PRESSURE_CONSTRAINT=%d\n"
               "- SFEM_RESTART_FOLDER=%s\n"
               "- SFEM_RESTART_ID=%d\n"
               "----------------------------------------\n",
               SFEM_DT,
               SFEM_CFL,
               SFEM_LUMPED_MASS,
               SFEM_VISCOSITY,
               SFEM_MASS_DENSITY,
               SFEM_VELOCITY_DIRICHLET_NODESET,
               SFEM_AVG_PRESSURE_CONSTRAINT,
               SFEM_RESTART_FOLDER,
               SFEM_RESTART_ID);
    }

    if (SFEM_RESTART_FOLDER && !SFEM_RESTART_ID) {
        fprintf(stderr, "Defined SFEM_RESTART_FOLDER but not SFEM_RESTART_ID\n");
        return EXIT_FAILURE;
    }

    real_t emin, emax;
    mesh_minmax_edge_length(&mesh, &emin, &emax);

    for (int s = 0; s < N_SYSTEMS; s++) {
        isolver_lsolve_set_max_iterations(&lsolve[s], SFEM_MAX_IT);
        isolver_lsolve_set_atol(&lsolve[s], SFEM_ATOL);
        isolver_lsolve_set_rtol(&lsolve[s], SFEM_RTOL);
        isolver_lsolve_set_stol(&lsolve[s], SFEM_STOL);
        isolver_lsolve_set_verbosity(&lsolve[s], SFEM_VERBOSE);
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
                              SFEM_PRESSURE_DIRICHLET_VALUE,
                              SFEM_PRESSURE_DIRICHLET_COMPONENT,
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

    for (int c = 0; c < n_pressure_dirichlet_conditions; c++) {
        pressure_dirichlet_conditions[c].local_size =
                remove_p2_nodes(pressure_dirichlet_conditions[c].local_size,
                                p1_nnodes,
                                pressure_dirichlet_conditions[c].idx);
    }

    ptrdiff_t p1_nnz = 0;
    count_t *p1_rowptr = 0;
    idx_t *p1_colidx = 0;
    build_crs_graph_for_elem_type(
            p1_type, mesh.nelements, p1_nnodes, mesh.elements, &p1_rowptr, &p1_colidx);
    p1_nnz = p1_rowptr[p1_nnodes];
    real_t *p1_values = calloc(p1_nnz, sizeof(real_t));

    laplacian_crs(p1_type,
                               mesh.nelements,
                               p1_nnodes,
                               mesh.elements,
                               mesh.points,
                               p1_rowptr,
                               p1_colidx,
                               p1_values);

    real_t *p1_mass_vector = 0;
    real_t sum_mass = 0;
    real_t *p1_inv_diag = 0;

    if (SFEM_AVG_PRESSURE_CONSTRAINT) {
        p1_mass_vector = calloc(p1_nnodes, sizeof(real_t));
        p1_inv_diag = calloc(p1_nnodes, sizeof(real_t));

        assemble_lumped_mass(
                p1_type, mesh.nelements, p1_nnodes, mesh.elements, mesh.points, p1_mass_vector);

        for (ptrdiff_t i = 0; i < p1_nnodes; i++) {
            sum_mass += p1_mass_vector[i];
        }

        constrained_gs_init(
                p1_nnodes, p1_rowptr, p1_colidx, p1_values, p1_mass_vector, p1_inv_diag);

    } else {
        for (int i = 0; i < n_pressure_dirichlet_conditions; i++) {
            crs_constraint_nodes_to_identity(pressure_dirichlet_conditions[i].local_size,
                                             pressure_dirichlet_conditions[i].idx,
                                             1,
                                             p1_rowptr,
                                             p1_colidx,
                                             p1_values);
        }

        // if (n_pressure_dirichlet_conditions == 0) {
        //     // make_matrix_nonsingular(
        //     //     SFEM_REGULARIZATION_FACTOR, p1_nnodes, p1_rowptr, p1_colidx, p1_values);

        //     p1_mass_vector = calloc(p1_nnodes, sizeof(real_t));

        //     assemble_lumped_mass(
        //         p1_type, mesh.nelements, p1_nnodes, mesh.elements, mesh.points, p1_mass_vector);

        //     shift_diag(p1_nnodes,
        //                SFEM_REGULARIZATION_FACTOR,
        //                p1_mass_vector,
        //                p1_rowptr,
        //                p1_colidx,
        //                p1_values);

        //     free(p1_mass_vector);
        // }

        isolver_lsolve_update_crs(&lsolve[INVERSE_POISSON_MATRIX],
                                  p1_nnodes,
                                  p1_nnodes,
                                  p1_rowptr,
                                  p1_colidx,
                                  p1_values);
    }

    ptrdiff_t p2_nnz = 0;
    count_t *p2_rowptr = 0;
    idx_t *p2_colidx = 0;
    real_t *p2_diffusion = 0;
    real_t *p2_mass_matrix = 0;

    int implicit_momentum = 0;

    if (!SFEM_LUMPED_MASS || implicit_momentum) {
        build_crs_graph_for_elem_type(mesh.element_type,
                                      mesh.nelements,
                                      mesh.nnodes,
                                      mesh.elements,
                                      &p2_rowptr,
                                      &p2_colidx);

        p2_nnz = p2_rowptr[mesh.nnodes];
        p2_mass_matrix = calloc(p2_nnz, sizeof(real_t));
    }

    if (implicit_momentum) {
        p2_diffusion = calloc(p2_nnz, sizeof(real_t));

        // Only works if B.C. are set on al three vector components
        {
            // Momentum step (implicit)
            navier_stokes_momentum_lhs_scalar_crs(mesh.element_type,
                                                  mesh.nelements,
                                                  mesh.nnodes,
                                                  mesh.elements,
                                                  mesh.points,
                                                  1,
                                                  SFEM_VISCOSITY,
                                                  p2_rowptr,
                                                  p2_colidx,
                                                  p2_diffusion);

            for (int i = 0; i < n_velocity_dirichlet_conditions; i++) {
                crs_constraint_nodes_to_identity(velocity_dirichlet_conditions[i].local_size,
                                                 velocity_dirichlet_conditions[i].idx,
                                                 1,
                                                 p2_rowptr,
                                                 p2_colidx,
                                                 p2_diffusion);
            }

            isolver_lsolve_update_crs(
                    &lsolve[1], mesh.nnodes, mesh.nnodes, p2_rowptr, p2_colidx, p2_diffusion);
        }
    }

    // Only works if B.C. are set on al three vector components
    if (!SFEM_LUMPED_MASS) {
        p2_nnz = p2_rowptr[mesh.nnodes];
        p2_mass_matrix = calloc(p2_nnz, sizeof(real_t));

        // Projection Step
        assemble_mass(mesh.element_type,
                      mesh.nelements,
                      mesh.nnodes,
                      mesh.elements,
                      mesh.points,
                      p2_rowptr,
                      p2_colidx,
                      p2_mass_matrix);

        // Seems that it works best when no B.C. are used here!
        // for (int i = 0; i < n_velocity_dirichlet_conditions; i++) {
        //     crs_constraint_nodes_to_identity(velocity_dirichlet_conditions[i].local_size,
        //                                      velocity_dirichlet_conditions[i].idx,
        //                                      1,
        //                                      p2_rowptr,
        //                                      p2_colidx,
        //                                      p2_mass_matrix);
        // }

        isolver_lsolve_update_crs(&lsolve[INVERSE_MASS_MATRIX],
                                  mesh.nnodes,
                                  mesh.nnodes,
                                  p2_rowptr,
                                  p2_colidx,
                                  p2_mass_matrix);
    } else {
        p2_mass_matrix = calloc(mesh.nnodes, sizeof(real_t));
        assemble_lumped_mass(mesh.element_type,
                             mesh.nelements,
                             mesh.nnodes,
                             mesh.elements,
                             mesh.points,
                             p2_mass_matrix);

        // array_write(comm, "out/mass.raw", SFEM_MPI_REAL_T, p2_mass_matrix, mesh.nnodes,
        // mesh.nnodes);
    }

    real_t *vel[3];
    real_t *correction[3];
    real_t *tentative_vel[3];
    real_t *buff = calloc(mesh.nnodes, sizeof(real_t));

    for (int d = 0; d < sdim; d++) {
        vel[d] = calloc(mesh.nnodes, sizeof(real_t));
        tentative_vel[d] = calloc(mesh.nnodes, sizeof(real_t));
        correction[d] = calloc(mesh.nnodes, sizeof(real_t));
    }

    real_t *p = calloc(p1_nnodes, sizeof(real_t));
    for (int i = 0; i < n_velocity_dirichlet_conditions; i++) {
        boundary_condition_t cond = velocity_dirichlet_conditions[i];
        if (cond.values) {
            constraint_nodes_to_values(cond.local_size, cond.idx, cond.values, vel[cond.component]);
        } else {
            constraint_nodes_to_value(cond.local_size, cond.idx, cond.value, vel[cond.component]);
        }
    }

    int export_counter = 0;
    real_t next_check_point = SFEM_EXPORT_FREQUENCY;
    logger_t time_logger;

    if (SFEM_RESTART_FOLDER) {
        for (int d = 0; d < sdim; d++) {
            sprintf(path, "%s/u%d.%09d.raw", SFEM_RESTART_FOLDER, d, SFEM_RESTART_ID);

            if (array_read(comm, path, SFEM_MPI_REAL_T, (void *)vel[d], mesh.nnodes, mesh.nnodes)) {
                fprintf(stderr, "Error reading restart file: %s\n", SFEM_RESTART_FOLDER);
                return EXIT_FAILURE;
            }
        }

        // Read current time file and start from offset
        // log_write_double()

        export_counter = SFEM_RESTART_ID + 1;

    } else {  // Write to disk
        printf("%g/%g\n", 0., SFEM_MAX_TIME);
        for (int d = 0; d < sdim; d++) {
            sprintf(path, "%s/u%d.%09d.raw", output_folder, d, export_counter);
            array_write(comm, path, SFEM_MPI_REAL_T, vel[d], mesh.nnodes, mesh.nnodes);
        }

        sprintf(path, "%s/p.%09d.raw", output_folder, export_counter);
        array_write(comm, path, SFEM_MPI_REAL_T, p, p1_nnodes, p1_nnodes);

        sprintf(path, "%s/div.%09d.raw", output_folder, export_counter);
        array_write(comm, path, SFEM_MPI_REAL_T, p, p1_nnodes, p1_nnodes);

        sprintf(path, "%s/div_pre.%09d.raw", output_folder, export_counter);
        array_write(comm, path, SFEM_MPI_REAL_T, p, p1_nnodes, p1_nnodes);

        sprintf(path, "%s/time.txt", output_folder);
        log_create_file(&time_logger, path, "w");
        log_write_double(&time_logger, 0);
        export_counter++;
    }

    real_t dt = SFEM_DT;
    ptrdiff_t step_count = 0;
    for (real_t t = 0; t < SFEM_MAX_TIME; t += dt, step_count++) {
        //////////////////////////////////////////////////////////////
        // Tentative momentum step
        //////////////////////////////////////////////////////////////
        {
            for (int d = 0; d < sdim; d++) {
                memset(correction[d], 0, mesh.nnodes * sizeof(real_t));
            }

            if (implicit_momentum) {
                // TODO
            } else {
                // Ensure CFL condition (Maybe not the right place)
                real_t max_velocity = 0;
                for (int d = 0; d < sdim; d++) {
                    for (ptrdiff_t i = 0; i < mesh.nnodes; i++) {
                        max_velocity = MAX(max_velocity, vel[d][i]);
                    }
                }

                dt = MAX(1e-12, MIN(SFEM_DT, SFEM_CFL / ((2 * max_velocity * emin * emin))));

                navier_stokes_mixed_explict_momentum_tentative(
                        mesh.element_type,
                        mesh.nelements,
                        mesh.nnodes,
                        mesh.elements,
                        mesh.points,
                        dt,
                        SFEM_VISCOSITY,
                        1,  // Turn-off convective term for debugging with 0
                        vel,
                        correction);

                {  // CHECK NaN
                    ptrdiff_t stop = 0;
                    for (int d = 0; d < sdim; d++) {
                        stop += count_nan(mesh.nnodes, correction[d]);
                    }

                    if (stop) {
                        fprintf(stderr, "Encountered %ld NaN value! Stopping...\n", stop);
                        MPI_Abort(comm, -1);
                        break;
                    }
                }

                for (int i = 0; i < n_velocity_dirichlet_conditions; i++) {
                    boundary_condition_t cond = velocity_dirichlet_conditions[i];
                    constraint_nodes_to_value(
                            cond.local_size, cond.idx, 0, correction[cond.component]);
                }

                for (int d = 0; d < sdim; d++) {
                    if (SFEM_LUMPED_MASS) {
#pragma omp parallel
                        {
#pragma omp for
                            for (ptrdiff_t i = 0; i < mesh.nnodes; i++) {
                                assert(correction[d][i] == correction[d][i]);
                                tentative_vel[d][i] = correction[d][i] / p2_mass_matrix[i];
                                assert(tentative_vel[d][i] == tentative_vel[d][i]);
                            }
                        }

                    } else {
                        memset(tentative_vel[d], 0, mesh.nnodes * sizeof(real_t));
                        isolver_lsolve_apply(
                                &lsolve[INVERSE_MASS_MATRIX], correction[d], tentative_vel[d]);
                    }

                    for (ptrdiff_t i = 0; i < mesh.nnodes; i++) {
                        tentative_vel[d][i] += vel[d][i];
                    }
                }
            }
        }

        //////////////////////////////////////////////////////////////
        // Poisson problem + solve
        //////////////////////////////////////////////////////////////
        {
            memset(buff, 0, p1_nnodes * sizeof(real_t));
            navier_stokes_mixed_divergence(mesh.element_type,
                                           p1_type,
                                           mesh.nelements,
                                           mesh.nnodes,
                                           mesh.elements,
                                           mesh.points,
                                           1,
                                           1,
                                           SFEM_VISCOSITY,
                                           tentative_vel,
                                           buff);

            for (int i = 0; i < n_pressure_dirichlet_conditions; i++) {
                boundary_condition_t cond = pressure_dirichlet_conditions[i];
                assert(cond.component == 0);
                constraint_nodes_to_value(cond.local_size, cond.idx, cond.value, buff);
            }

            if (t >= next_check_point) {  // Write to disk
                memset(p, 0, p1_nnodes * sizeof(real_t));

                apply_inv_lumped_mass(
                        p1_type, mesh.nelements, p1_nnodes, mesh.elements, mesh.points, buff, p);

                sprintf(path, "%s/div_pre.%09d.raw", output_folder, export_counter);
                array_write(comm, path, SFEM_MPI_REAL_T, p, p1_nnodes, p1_nnodes);
            }

            memset(p, 0, p1_nnodes * sizeof(real_t));

            if (SFEM_AVG_PRESSURE_CONSTRAINT) {
                real_t lagrange_multiplier = 0;

                int check_each = 100;
                for (long i = 0; i * check_each < SFEM_MAX_IT; i++) {
                    constrained_gs(p1_nnodes,
                                   p1_rowptr,
                                   p1_colidx,
                                   p1_values,
                                   p1_inv_diag,
                                   buff,
                                   p,
                                   p1_mass_vector,
                                   sum_mass,
                                   &lagrange_multiplier,
                                   check_each);

                    real_t res = 0;
                    constrained_gs_residual(p1_nnodes,
                                            p1_rowptr,
                                            p1_colidx,
                                            p1_values,
                                            buff,
                                            p,
                                            p1_mass_vector,
                                            sum_mass,
                                            lagrange_multiplier,
                                            &res);

                    printf("(%ld) poisson residual: %g, lagrange_multiplier: %g\n",
                           (i + 1) * check_each,
                           res,
                           lagrange_multiplier);

                    if (res < SFEM_ATOL) break;
                }
            } else {
                isolver_lsolve_apply(&lsolve[INVERSE_POISSON_MATRIX], buff, p);
            }
        }
        //////////////////////////////////////////////////////////////
        // Correction/Projection step
        //////////////////////////////////////////////////////////////
        {
            for (int d = 0; d < sdim; d++) {
                memset(correction[d], 0, mesh.nnodes * sizeof(real_t));
            }

            navier_stokes_mixed_correction(mesh.element_type,
                                           p1_type,
                                           mesh.nelements,
                                           mesh.nnodes,
                                           mesh.elements,
                                           mesh.points,
                                           1,
                                           1,
                                           p,
                                           correction);

            for (int d = 0; d < sdim; d++) {
                if (SFEM_LUMPED_MASS) {
#pragma omp parallel
                    {
#pragma omp for
                        for (ptrdiff_t i = 0; i < mesh.nnodes; i++) {
                            correction[d][i] = correction[d][i] / p2_mass_matrix[i];
                            assert(correction[d][i] == correction[d][i]);
                        }
                    }

                } else {
                    memset(buff, 0, mesh.nnodes * sizeof(real_t));
                    isolver_lsolve_apply(&lsolve[INVERSE_MASS_MATRIX], correction[d], buff);
                    memcpy(correction[d], buff, mesh.nnodes * sizeof(real_t));
                }
            }

            for (int d = 0; d < sdim; d++) {
                for (ptrdiff_t i = 0; i < mesh.nnodes; i++) {
                    vel[d][i] = tentative_vel[d][i] + correction[d][i];
                }
            }

            for (int i = 0; i < n_velocity_dirichlet_conditions; i++) {
                boundary_condition_t cond = velocity_dirichlet_conditions[i];
                if (cond.values) {
                    constraint_nodes_to_values(
                            cond.local_size, cond.idx, cond.values, vel[cond.component]);
                } else {
                    constraint_nodes_to_value(
                            cond.local_size, cond.idx, cond.value, vel[cond.component]);
                }
            }
        }

        if (t >= next_check_point) {  // Write to disk
            printf("%g/%g dt=%g\n", t, SFEM_MAX_TIME, dt);
            for (int d = 0; d < sdim; d++) {
                sprintf(path, "%s/u%d.%09d.raw", output_folder, d, export_counter);
                array_write(comm, path, SFEM_MPI_REAL_T, vel[d], mesh.nnodes, mesh.nnodes);
            }

            sprintf(path, "%s/p.%09d.raw", output_folder, export_counter);
            array_write(comm, path, SFEM_MPI_REAL_T, p, p1_nnodes, p1_nnodes);

            {  // Compute divergence for analysis
                memset(buff, 0, p1_nnodes * sizeof(real_t));
                navier_stokes_mixed_divergence(mesh.element_type,
                                               p1_type,
                                               mesh.nelements,
                                               mesh.nnodes,
                                               mesh.elements,
                                               mesh.points,
                                               1,
                                               1,
                                               SFEM_VISCOSITY,
                                               vel,
                                               buff);

                memset(p, 0, p1_nnodes * sizeof(real_t));
                apply_inv_lumped_mass(
                        p1_type, mesh.nelements, p1_nnodes, mesh.elements, mesh.points, buff, p);

                sprintf(path, "%s/div.%09d.raw", output_folder, export_counter);
                array_write(comm, path, SFEM_MPI_REAL_T, p, p1_nnodes, p1_nnodes);

                log_write_double(&time_logger, t);
            }

            next_check_point += SFEM_EXPORT_FREQUENCY;
            export_counter++;
        }
    }

    // Free resources
    free(p1_rowptr);
    free(p1_colidx);
    free(p1_values);

    free(p2_rowptr);
    free(p2_colidx);
    free(p2_diffusion);
    free(p2_mass_matrix);

    free(p);
    free(buff);

    for (int d = 0; d < sdim; d++) {
        free(vel[d]);
        free(tentative_vel[d]);
        free(correction[d]);
    }

    if (p1_mass_vector) {
        free(p1_mass_vector);
    }

    ptrdiff_t nelements = mesh.nelements;
    ptrdiff_t nnodes = mesh.nnodes;

    mesh_destroy(&mesh);

    destroy_conditions(n_velocity_dirichlet_conditions, velocity_dirichlet_conditions);
    destroy_conditions(n_pressure_dirichlet_conditions, pressure_dirichlet_conditions);
    // destroy_conditions(n_neumann_conditions, neumann_conditions);

    log_destroy(&time_logger);

    double tock = MPI_Wtime();
    if (!rank) {
        printf("----------------------------------------\n");
        printf("#elements %ld #nodes %ld #nz %ld #steps %ld\n",
               (long)nelements,
               (long)nnodes,
               (long)p2_nnz,
               step_count);
        printf("TTS:\t\t\t%g seconds\n", tock - tick);
    }

    // FIXME One is enough for now, but it is not clean
    isolver_lsolve_destroy(&lsolve[INVERSE_POISSON_MATRIX]);
    // isolver_lsolve_destroy(&lsolve[1]);
    return MPI_Finalize();
}
