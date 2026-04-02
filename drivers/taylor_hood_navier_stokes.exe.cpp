#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <limits>

#include "array_dtof.h"
#include "matrixio_array.h"
#include "matrixio_crs.h"
#include "utils.h"

#include "sfem_base.hpp"
#include "sfem_defs.hpp"
#include "sfem_vec.hpp"
#include "sortreduce.hpp"

#include "mass.hpp"

#include "boundary_condition.hpp"
#include "boundary_condition_io.hpp"
#include "dirichlet.hpp"
#include "neumann.hpp"

#include "laplacian.hpp"
#include "navier_stokes.hpp"

#include "constrained_gs.hpp"

#include "sfem_API.hpp"
#include "sfem_ShiftableJacobi.hpp"
#include "sfem_bcgs.hpp"
#include "sfem_context.hpp"
#include "sfem_crs_SpMV.hpp"
#include "sfem_logger.hpp"
#include "smesh_output.hpp"
#include "spmv.hpp"

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

static void make_matrix_nonsingular(const real_t                       factor,
                                    const ptrdiff_t                    n,
                                    const count_t *const SFEM_RESTRICT rowptr,
                                    const idx_t *const SFEM_RESTRICT   colidx,
                                    real_t *const SFEM_RESTRICT        values) {
    count_t            rstart  = rowptr[n - 1];
    count_t            rextent = rowptr[n] - rstart;
    const idx_t *const cols    = &colidx[rstart];
    real_t *const      vals    = &values[rstart];

    for (count_t i = 0; i < rextent; ++i) {
        if (cols[i] == n - 1) {
            vals[i] *= (1 + factor);

            // vals[i] += factor;
            break;
        }
    }
}

static void shift_diag(const ptrdiff_t                    n,
                       const real_t                       factor,
                       const real_t *const SFEM_RESTRICT  vector,
                       const count_t *const SFEM_RESTRICT rowptr,
                       const idx_t *const SFEM_RESTRICT   colidx,
                       real_t *const SFEM_RESTRICT        values) {
    for (ptrdiff_t i = 0; i < n; ++i) {
        // ptrdiff_t i = n - 1;
        count_t            rstart  = rowptr[i];
        count_t            rextent = rowptr[i + 1] - rstart;
        const idx_t *const cols    = &colidx[rstart];
        real_t *const      vals    = &values[rstart];

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

static std::shared_ptr<sfem::BiCGStab<real_t>> make_crs_bcgs_solver(const ptrdiff_t                    n,
                                                                    const sfem::SharedBuffer<count_t> &rowptr,
                                                                    const sfem::SharedBuffer<idx_t>   &colidx,
                                                                    const sfem::SharedBuffer<real_t>  &values,
                                                                    const int                          max_it,
                                                                    const real_t                       tol,
                                                                    const bool                         verbose) {
    auto op = sfem::h_crs_spmv<count_t, idx_t, real_t>(n, n, rowptr, colidx, values, 0);

    auto diag = sfem::create_host_buffer<real_t>(n);
    crs_diag(n, rowptr->data(), colidx->data(), values->data(), diag->data());

    auto solver = sfem::h_bcgs<real_t>();
    solver->set_n_dofs(n);
    solver->set_max_it(max_it);
    solver->set_atol(tol);
    solver->verbose = verbose;
    solver->set_op(op);
    solver->set_preconditioner_op(sfem::h_shiftable_jacobi(diag));
    return solver;
}

static void mesh_minmax_edge_length(const std::shared_ptr<sfem::Mesh> &mesh, real_t *emin, real_t *emax) {
    auto graph  = mesh->node_to_node_graph_upper_triangular();
    auto rowptr = graph->rowptr()->data();
    auto colidx = graph->colidx()->data();
    auto points = mesh->points()->data();

    real_t local_min = std::numeric_limits<real_t>::max();
    real_t local_max = 0;

    for (ptrdiff_t i = 0; i < mesh->n_nodes(); ++i) {
        for (count_t k = rowptr[i]; k < rowptr[i + 1]; ++k) {
            const idx_t j = colidx[k];

            real_t len2 = 0;
            for (int d = 0; d < mesh->spatial_dimension(); ++d) {
                const real_t diff = points[d][j] - points[d][i];
                len2 += diff * diff;
            }

            const real_t len = sqrt(len2);
            local_min        = MIN(local_min, len);
            local_max        = MAX(local_max, len);
        }
    }

    MPI_Allreduce(&local_min, emin, 1, SFEM_MPI_REAL_T, MPI_MIN, mesh->comm()->get());
    MPI_Allreduce(&local_max, emax, 1, SFEM_MPI_REAL_T, MPI_MAX, mesh->comm()->get());
}

static int write_output_step(const std::shared_ptr<smesh::Output> &output,
                             const char                           *field_name,
                             const int                             step,
                             const real_t                         *data) {
    char buffer[SFEM_MAX_PATH_LENGTH];
    snprintf(buffer, sizeof(buffer), "%s.%09d", field_name, step);
    return output->write_nodal(buffer, smesh::TypeToEnum<real_t>::value(), data);
}

int main(int argc, char *argv[]) {
    auto ctx = sfem::initialize_serial(argc, argv);

    MPI_Comm                                comm = ctx->communicator()->get();
    std::shared_ptr<sfem::BiCGStab<real_t>> solver[N_SYSTEMS];

    const int rank = ctx->communicator()->rank();
    const int size = ctx->communicator()->size();

    if (size != 1) {
        fprintf(stderr, "Parallel execution not supported!\n");
        return EXIT_FAILURE;
    }

    if (argc != 3) {
        fprintf(stderr, "usage: %s <folder> <output>\n", argv[0]);
        return EXIT_FAILURE;
    }

    const char *output_folder = argv[2];

    char   path[SFEM_MAX_PATH_LENGTH];
    double tick = MPI_Wtime();

    ///////////////////////////////////////////////////////////////////////////////
    // Read data
    ///////////////////////////////////////////////////////////////////////////////

    const char *folder = argv[1];

    auto mesh = sfem::Mesh::create_from_file(sfem::Communicator::wrap(comm), smesh::Path(folder));
    if (mesh->n_blocks() != 1) {
        fprintf(stderr, "%s requires a single mesh block\n", argv[0]);
        return EXIT_FAILURE;
    }

    const auto mesh_element_type = mesh->element_type(0);
    const auto mesh_nelements    = mesh->n_elements();
    const auto mesh_nnodes       = mesh->n_nodes();
    const int  mesh_spatial_dim  = mesh->spatial_dimension();
    auto       mesh_elements     = mesh->elements(0)->data();
    auto       mesh_points       = mesh->points()->data();
    auto       velocity_output   = smesh::Output::create(mesh, smesh::Path(output_folder));

    const real_t SFEM_VISCOSITY    = smesh::Env::read("SFEM_VISCOSITY", real_t(1));
    const real_t SFEM_MASS_DENSITY = smesh::Env::read("SFEM_MASS_DENSITY", real_t(1));

    const std::string SFEM_VELOCITY_DIRICHLET_NODESET_str   = smesh::Env::read_string("SFEM_VELOCITY_DIRICHLET_NODESET", "");
    const std::string SFEM_VELOCITY_DIRICHLET_VALUE_str     = smesh::Env::read_string("SFEM_VELOCITY_DIRICHLET_VALUE", "");
    const std::string SFEM_VELOCITY_DIRICHLET_COMPONENT_str = smesh::Env::read_string("SFEM_VELOCITY_DIRICHLET_COMPONENT", "");
    const char       *SFEM_VELOCITY_DIRICHLET_NODESET =
            SFEM_VELOCITY_DIRICHLET_NODESET_str.empty() ? nullptr : SFEM_VELOCITY_DIRICHLET_NODESET_str.c_str();
    const char *SFEM_VELOCITY_DIRICHLET_VALUE =
            SFEM_VELOCITY_DIRICHLET_VALUE_str.empty() ? nullptr : SFEM_VELOCITY_DIRICHLET_VALUE_str.c_str();
    const char *SFEM_VELOCITY_DIRICHLET_COMPONENT =
            SFEM_VELOCITY_DIRICHLET_COMPONENT_str.empty() ? nullptr : SFEM_VELOCITY_DIRICHLET_COMPONENT_str.c_str();

    const std::string SFEM_PRESSURE_DIRICHLET_NODESET_str   = smesh::Env::read_string("SFEM_PRESSURE_DIRICHLET_NODESET", "");
    const std::string SFEM_PRESSURE_DIRICHLET_VALUE_str     = smesh::Env::read_string("SFEM_PRESSURE_DIRICHLET_VALUE", "");
    const std::string SFEM_PRESSURE_DIRICHLET_COMPONENT_str = smesh::Env::read_string("SFEM_PRESSURE_DIRICHLET_COMPONENT", "");
    const char       *SFEM_PRESSURE_DIRICHLET_NODESET =
            SFEM_PRESSURE_DIRICHLET_NODESET_str.empty() ? nullptr : SFEM_PRESSURE_DIRICHLET_NODESET_str.c_str();
    const char *SFEM_PRESSURE_DIRICHLET_VALUE =
            SFEM_PRESSURE_DIRICHLET_VALUE_str.empty() ? nullptr : SFEM_PRESSURE_DIRICHLET_VALUE_str.c_str();
    const char *SFEM_PRESSURE_DIRICHLET_COMPONENT =
            SFEM_PRESSURE_DIRICHLET_COMPONENT_str.empty() ? nullptr : SFEM_PRESSURE_DIRICHLET_COMPONENT_str.c_str();

    const int    SFEM_MAX_IT  = smesh::Env::read("SFEM_MAX_IT", 100);
    const real_t SFEM_ATOL    = smesh::Env::read("SFEM_ATOL", real_t(1e-15));
    const real_t SFEM_RTOL    = smesh::Env::read("SFEM_RTOL", real_t(1e-14));
    const real_t SFEM_STOL    = smesh::Env::read("SFEM_STOL", real_t(1e-12));
    const int    SFEM_VERBOSE = smesh::Env::read("SFEM_VERBOSE", 0);

    const real_t SFEM_DT               = smesh::Env::read("SFEM_DT", real_t(1));
    const real_t SFEM_MAX_TIME         = smesh::Env::read("SFEM_MAX_TIME", real_t(1));
    const real_t SFEM_CFL              = smesh::Env::read("SFEM_CFL", real_t(0.1));
    const real_t SFEM_EXPORT_FREQUENCY = smesh::Env::read("SFEM_EXPORT_FREQUENCY", real_t(0.1));
    const int    SFEM_LUMPED_MASS      = smesh::Env::read("SFEM_LUMPED_MASS", 0);

    const int    SFEM_AVG_PRESSURE_CONSTRAINT = smesh::Env::read("SFEM_AVG_PRESSURE_CONSTRAINT", 0);
    const real_t SFEM_REGULARIZATION_FACTOR   = smesh::Env::read("SFEM_REGULARIZATION_FACTOR", real_t(1e-8));

    const std::string SFEM_RESTART_FOLDER_str = smesh::Env::read_string("SFEM_RESTART_FOLDER", "");
    const char       *SFEM_RESTART_FOLDER     = SFEM_RESTART_FOLDER_str.empty() ? nullptr : SFEM_RESTART_FOLDER_str.c_str();
    const int         SFEM_RESTART_ID         = smesh::Env::read("SFEM_RESTART_ID", 0);

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
               SFEM_VELOCITY_DIRICHLET_NODESET ? SFEM_VELOCITY_DIRICHLET_NODESET : "(null)",
               SFEM_AVG_PRESSURE_CONSTRAINT,
               SFEM_RESTART_FOLDER ? SFEM_RESTART_FOLDER : "(null)",
               SFEM_RESTART_ID);
    }

    if (SFEM_RESTART_FOLDER && !SFEM_RESTART_ID) {
        fprintf(stderr, "Defined SFEM_RESTART_FOLDER but not SFEM_RESTART_ID\n");
        return EXIT_FAILURE;
    }

    real_t emin, emax;
    mesh_minmax_edge_length(mesh, &emin, &emax);
    const real_t linear_solver_tol = MAX(SFEM_STOL, MAX(SFEM_ATOL, SFEM_RTOL));

    // int n_neumann_conditions;
    // boundary_condition_t *neumann_conditions;

    int                   n_velocity_dirichlet_conditions;
    boundary_condition_t *velocity_dirichlet_conditions;

    int                   n_pressure_dirichlet_conditions;
    boundary_condition_t *pressure_dirichlet_conditions;

    read_boundary_conditions(comm,
                             SFEM_VELOCITY_DIRICHLET_NODESET,
                             SFEM_VELOCITY_DIRICHLET_VALUE,
                             SFEM_VELOCITY_DIRICHLET_COMPONENT,
                             &velocity_dirichlet_conditions,
                             &n_velocity_dirichlet_conditions);

    read_boundary_conditions(comm,
                             SFEM_PRESSURE_DIRICHLET_NODESET,
                             SFEM_PRESSURE_DIRICHLET_VALUE,
                             SFEM_PRESSURE_DIRICHLET_COMPONENT,
                             &pressure_dirichlet_conditions,
                             &n_pressure_dirichlet_conditions);

    smesh::ElemType p1_type   = elem_lower_order(mesh_element_type);
    const int       p1_nxe    = elem_num_nodes(p1_type);
    const int       sdim      = elem_manifold_dim(mesh_element_type);
    ptrdiff_t       p1_nnodes = 0;

    for (int d = 0; d < p1_nxe; d++) {
        p1_nnodes = MAX(p1_nnodes, max_idx(mesh_nelements, mesh_elements[d]));
    }

    p1_nnodes += 1;

    for (int c = 0; c < n_pressure_dirichlet_conditions; c++) {
        pressure_dirichlet_conditions[c].local_size =
                remove_p2_nodes(pressure_dirichlet_conditions[c].local_size, p1_nnodes, pressure_dirichlet_conditions[c].idx);
    }

    ptrdiff_t p1_nnz    = 0;
    count_t  *p1_rowptr = 0;
    idx_t    *p1_colidx = 0;
    build_crs_graph_for_elem_type(p1_type, mesh_nelements, p1_nnodes, mesh_elements, &p1_rowptr, &p1_colidx);
    p1_nnz                   = p1_rowptr[p1_nnodes];
    real_t *p1_values        = (real_t *)calloc(p1_nnz, sizeof(real_t));
    auto    p1_rowptr_buffer = sfem::manage_host_buffer<count_t>(p1_nnodes + 1, p1_rowptr);
    auto    p1_colidx_buffer = sfem::manage_host_buffer<idx_t>(p1_nnz, p1_colidx);
    auto    p1_values_buffer = sfem::manage_host_buffer<real_t>(p1_nnz, p1_values);
    auto    p1_mesh          = std::make_shared<sfem::Mesh>(mesh->comm(),
                                                p1_type,
                                                sfem::view(mesh->elements(0), 0, p1_nxe, 0, mesh_nelements),
                                                sfem::view(mesh->points(), 0, mesh_spatial_dim, 0, p1_nnodes));
    auto    pressure_output  = smesh::Output::create(p1_mesh, smesh::Path(output_folder));

    laplacian_crs(p1_type, mesh_nelements, p1_nnodes, mesh_elements, mesh_points, p1_rowptr, p1_colidx, p1_values);

    real_t *p1_mass_vector = 0;
    real_t  sum_mass       = 0;
    real_t *p1_inv_diag    = 0;

    if (SFEM_AVG_PRESSURE_CONSTRAINT) {
        p1_mass_vector = (real_t *)calloc(p1_nnodes, sizeof(real_t));
        p1_inv_diag    = (real_t *)calloc(p1_nnodes, sizeof(real_t));

        assemble_lumped_mass(p1_type, mesh_nelements, p1_nnodes, mesh_elements, mesh_points, p1_mass_vector);

        for (ptrdiff_t i = 0; i < p1_nnodes; i++) {
            sum_mass += p1_mass_vector[i];
        }

        constrained_gs_init(p1_nnodes, p1_rowptr, p1_colidx, p1_values, p1_mass_vector, p1_inv_diag);

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

        //     shift_diag(p1_nnodes,
        //                SFEM_REGULARIZATION_FACTOR,
        //                p1_mass_vector,
        //                p1_rowptr,
        //                p1_colidx,
        //                p1_values);

        //     free(p1_mass_vector);
        // }

        solver[INVERSE_POISSON_MATRIX] = make_crs_bcgs_solver(
                p1_nnodes, p1_rowptr_buffer, p1_colidx_buffer, p1_values_buffer, SFEM_MAX_IT, linear_solver_tol, SFEM_VERBOSE);
    }

    ptrdiff_t                   p2_nnz         = 0;
    count_t                    *p2_rowptr      = 0;
    idx_t                      *p2_colidx      = 0;
    real_t                     *p2_diffusion   = 0;
    real_t                     *p2_mass_matrix = 0;
    sfem::SharedBuffer<count_t> p2_rowptr_buffer;
    sfem::SharedBuffer<idx_t>   p2_colidx_buffer;
    sfem::SharedBuffer<real_t>  p2_diffusion_buffer;
    sfem::SharedBuffer<real_t>  p2_mass_matrix_buffer;

    int implicit_momentum = 0;

    if (!SFEM_LUMPED_MASS || implicit_momentum) {
        build_crs_graph_for_elem_type(mesh_element_type, mesh_nelements, mesh_nnodes, mesh_elements, &p2_rowptr, &p2_colidx);

        p2_nnz           = p2_rowptr[mesh_nnodes];
        p2_rowptr_buffer = sfem::manage_host_buffer<count_t>(mesh_nnodes + 1, p2_rowptr);
        p2_colidx_buffer = sfem::manage_host_buffer<idx_t>(p2_nnz, p2_colidx);
    }

    if (implicit_momentum) {
        p2_diffusion        = (real_t *)calloc(p2_nnz, sizeof(real_t));
        p2_diffusion_buffer = sfem::manage_host_buffer<real_t>(p2_nnz, p2_diffusion);

        // Only works if B.C. are set on al three vector components
        {
            // Momentum step (implicit)
            navier_stokes_momentum_lhs_scalar_crs(mesh_element_type,
                                                  mesh_nelements,
                                                  mesh_nnodes,
                                                  mesh_elements,
                                                  mesh_points,
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

            solver[1] = make_crs_bcgs_solver(mesh_nnodes,
                                             p2_rowptr_buffer,
                                             p2_colidx_buffer,
                                             p2_diffusion_buffer,
                                             SFEM_MAX_IT,
                                             linear_solver_tol,
                                             SFEM_VERBOSE);
        }
    }

    // Only works if B.C. are set on al three vector components
    if (!SFEM_LUMPED_MASS) {
        p2_mass_matrix        = (real_t *)calloc(p2_nnz, sizeof(real_t));
        p2_mass_matrix_buffer = sfem::manage_host_buffer<real_t>(p2_nnz, p2_mass_matrix);

        // Projection Step
        assemble_mass(
                mesh_element_type, mesh_nelements, mesh_nnodes, mesh_elements, mesh_points, p2_rowptr, p2_colidx, p2_mass_matrix);

        // Seems that it works best when no B.C. are used here!
        // for (int i = 0; i < n_velocity_dirichlet_conditions; i++) {
        //     crs_constraint_nodes_to_identity(velocity_dirichlet_conditions[i].local_size,
        //                                      velocity_dirichlet_conditions[i].idx,
        //                                      1,
        //                                      p2_rowptr,
        //                                      p2_colidx,
        //                                      p2_mass_matrix);
        // }

        solver[INVERSE_MASS_MATRIX] = make_crs_bcgs_solver(mesh_nnodes,
                                                           p2_rowptr_buffer,
                                                           p2_colidx_buffer,
                                                           p2_mass_matrix_buffer,
                                                           SFEM_MAX_IT,
                                                           linear_solver_tol,
                                                           SFEM_VERBOSE);
    } else {
        p2_mass_matrix = (real_t *)calloc(mesh_nnodes, sizeof(real_t));
        assemble_lumped_mass(mesh_element_type, mesh_nelements, mesh_nnodes, mesh_elements, mesh_points, p2_mass_matrix);
    }

    real_t *vel[3];
    real_t *correction[3];
    real_t *tentative_vel[3];
    real_t *buff = (real_t *)calloc(mesh_nnodes, sizeof(real_t));

    for (int d = 0; d < sdim; d++) {
        vel[d]           = (real_t *)calloc(mesh_nnodes, sizeof(real_t));
        tentative_vel[d] = (real_t *)calloc(mesh_nnodes, sizeof(real_t));
        correction[d]    = (real_t *)calloc(mesh_nnodes, sizeof(real_t));
    }

    real_t *p = (real_t *)calloc(p1_nnodes, sizeof(real_t));
    for (int i = 0; i < n_velocity_dirichlet_conditions; i++) {
        boundary_condition_t cond = velocity_dirichlet_conditions[i];
        if (cond.values) {
            constraint_nodes_to_values(cond.local_size, cond.idx, cond.values, vel[cond.component]);
        } else {
            constraint_nodes_to_value(cond.local_size, cond.idx, cond.value, vel[cond.component]);
        }
    }

    int      export_counter   = 0;
    real_t   next_check_point = SFEM_EXPORT_FREQUENCY;
    logger_t time_logger;
    log_init(&time_logger);

    if (SFEM_RESTART_FOLDER) {
        for (int d = 0; d < sdim; d++) {
            sprintf(path, "%s/u%d.%09d.raw", SFEM_RESTART_FOLDER, d, SFEM_RESTART_ID);

            if (array_read(comm, path, SFEM_MPI_REAL_T, (void *)vel[d], mesh_nnodes, mesh_nnodes)) {
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
            snprintf(path, sizeof(path), "u%d", d);
            write_output_step(velocity_output, path, export_counter, vel[d]);
        }

        write_output_step(pressure_output, "p", export_counter, p);
        write_output_step(pressure_output, "div", export_counter, p);
        write_output_step(pressure_output, "div_pre", export_counter, p);

        sprintf(path, "%s/time.txt", output_folder);
        log_create_file(&time_logger, path, "w");
        log_write_double(&time_logger, 0);
        export_counter++;
    }

    real_t    dt         = SFEM_DT;
    ptrdiff_t step_count = 0;
    for (real_t t = 0; t < SFEM_MAX_TIME; t += dt, step_count++) {
        //////////////////////////////////////////////////////////////
        // Tentative momentum step
        //////////////////////////////////////////////////////////////
        {
            for (int d = 0; d < sdim; d++) {
                memset(correction[d], 0, mesh_nnodes * sizeof(real_t));
            }

            if (implicit_momentum) {
                // TODO
            } else {
                // Ensure CFL condition (Maybe not the right place)
                real_t max_velocity = 0;
                for (int d = 0; d < sdim; d++) {
                    for (ptrdiff_t i = 0; i < mesh_nnodes; i++) {
                        max_velocity = MAX(max_velocity, vel[d][i]);
                    }
                }

                dt = MAX(1e-12, MIN(SFEM_DT, SFEM_CFL / ((2 * max_velocity * emin * emin))));

                navier_stokes_mixed_explict_momentum_tentative(mesh_element_type,
                                                               mesh_nelements,
                                                               mesh_nnodes,
                                                               mesh_elements,
                                                               mesh_points,
                                                               dt,
                                                               SFEM_VISCOSITY,
                                                               1,  // Turn-off convective term for debugging with 0
                                                               vel,
                                                               correction);

                {  // CHECK NaN
                    ptrdiff_t stop = 0;
                    for (int d = 0; d < sdim; d++) {
                        stop += count_nan(mesh_nnodes, correction[d]);
                    }

                    if (stop) {
                        SFEM_ERROR("Encountered %ld NaN value! Stopping...\n", stop);
                        break;
                    }
                }

                for (int i = 0; i < n_velocity_dirichlet_conditions; i++) {
                    boundary_condition_t cond = velocity_dirichlet_conditions[i];
                    constraint_nodes_to_value(cond.local_size, cond.idx, 0, correction[cond.component]);
                }

                for (int d = 0; d < sdim; d++) {
                    if (SFEM_LUMPED_MASS) {
#pragma omp parallel
                        {
#pragma omp for
                            for (ptrdiff_t i = 0; i < mesh_nnodes; i++) {
                                assert(correction[d][i] == correction[d][i]);
                                tentative_vel[d][i] = correction[d][i] / p2_mass_matrix[i];
                                assert(tentative_vel[d][i] == tentative_vel[d][i]);
                            }
                        }

                    } else {
                        memset(tentative_vel[d], 0, mesh_nnodes * sizeof(real_t));
                        solver[INVERSE_MASS_MATRIX]->apply(correction[d], tentative_vel[d]);
                    }

                    for (ptrdiff_t i = 0; i < mesh_nnodes; i++) {
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
            navier_stokes_mixed_divergence(mesh_element_type,
                                           p1_type,
                                           mesh_nelements,
                                           mesh_nnodes,
                                           mesh_elements,
                                           mesh_points,
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

                apply_inv_lumped_mass(p1_type, mesh_nelements, p1_nnodes, mesh_elements, mesh_points, buff, p);
                write_output_step(pressure_output, "div_pre", export_counter, p);
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
                solver[INVERSE_POISSON_MATRIX]->apply(buff, p);
            }
        }
        //////////////////////////////////////////////////////////////
        // Correction/Projection step
        //////////////////////////////////////////////////////////////
        {
            for (int d = 0; d < sdim; d++) {
                memset(correction[d], 0, mesh_nnodes * sizeof(real_t));
            }

            navier_stokes_mixed_correction(
                    mesh_element_type, p1_type, mesh_nelements, mesh_nnodes, mesh_elements, mesh_points, 1, 1, p, correction);

            for (int d = 0; d < sdim; d++) {
                if (SFEM_LUMPED_MASS) {
#pragma omp parallel
                    {
#pragma omp for
                        for (ptrdiff_t i = 0; i < mesh_nnodes; i++) {
                            correction[d][i] = correction[d][i] / p2_mass_matrix[i];
                            assert(correction[d][i] == correction[d][i]);
                        }
                    }

                } else {
                    memset(buff, 0, mesh_nnodes * sizeof(real_t));
                    solver[INVERSE_MASS_MATRIX]->apply(correction[d], buff);
                    memcpy(correction[d], buff, mesh_nnodes * sizeof(real_t));
                }
            }

            for (int d = 0; d < sdim; d++) {
                for (ptrdiff_t i = 0; i < mesh_nnodes; i++) {
                    vel[d][i] = tentative_vel[d][i] + correction[d][i];
                }
            }

            for (int i = 0; i < n_velocity_dirichlet_conditions; i++) {
                boundary_condition_t cond = velocity_dirichlet_conditions[i];
                if (cond.values) {
                    constraint_nodes_to_values(cond.local_size, cond.idx, cond.values, vel[cond.component]);
                } else {
                    constraint_nodes_to_value(cond.local_size, cond.idx, cond.value, vel[cond.component]);
                }
            }
        }

        if (t >= next_check_point) {  // Write to disk
            printf("%g/%g dt=%g\n", t, SFEM_MAX_TIME, dt);
            for (int d = 0; d < sdim; d++) {
                snprintf(path, sizeof(path), "u%d", d);
                write_output_step(velocity_output, path, export_counter, vel[d]);
            }

            write_output_step(pressure_output, "p", export_counter, p);

            {  // Compute divergence for analysis
                memset(buff, 0, p1_nnodes * sizeof(real_t));
                navier_stokes_mixed_divergence(mesh_element_type,
                                               p1_type,
                                               mesh_nelements,
                                               mesh_nnodes,
                                               mesh_elements,
                                               mesh_points,
                                               1,
                                               1,
                                               SFEM_VISCOSITY,
                                               vel,
                                               buff);

                memset(p, 0, p1_nnodes * sizeof(real_t));
                apply_inv_lumped_mass(p1_type, mesh_nelements, p1_nnodes, mesh_elements, mesh_points, buff, p);
                write_output_step(pressure_output, "div", export_counter, p);

                log_write_double(&time_logger, t);
            }

            next_check_point += SFEM_EXPORT_FREQUENCY;
            export_counter++;
        }
    }

    // Free resources
    if (SFEM_LUMPED_MASS) {
        free(p2_mass_matrix);
    }

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

    if (p1_inv_diag) {
        free(p1_inv_diag);
    }

    ptrdiff_t nelements = mesh_nelements;
    ptrdiff_t nnodes    = mesh_nnodes;

    destroy_conditions(n_velocity_dirichlet_conditions, velocity_dirichlet_conditions);
    destroy_conditions(n_pressure_dirichlet_conditions, pressure_dirichlet_conditions);
    // destroy_conditions(n_neumann_conditions, neumann_conditions);

    log_destroy(&time_logger);

    double tock = MPI_Wtime();
    if (!rank) {
        printf("----------------------------------------\n");
        printf("#elements %ld #nodes %ld #nz %ld #steps %ld\n", (long)nelements, (long)nnodes, (long)p2_nnz, step_count);
        printf("TTS:\t\t\t%g seconds\n", tock - tick);
    }

    return EXIT_SUCCESS;
}
