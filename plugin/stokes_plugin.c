#include "isolver_function.h"

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#include "array_dtof.h"
#include "matrixio_array.h"
#include "matrixio_crs.h"
#include "utils.h"

#include "crs_graph.h"
#include "sfem_base.h"

#include "stokes_mini.h"

#include "read_mesh.h"
#include "sfem_defs.h"
#include "sfem_mesh.h"

#include "boundary_condition.h"
#include "dirichlet.h"
#include "neumann.h"

typedef struct {
    mesh_t *mesh;

    count_t *n2n_rowptr;
    idx_t *n2n_colidx;
    int block_size;

    int n_neumann_conditions;
    boundary_condition_t *neumann_conditions;

    int n_dirichlet_conditions;
    boundary_condition_t *dirichlet_conditions;

    real_t mu, rho;

    const char *output_dir;
    const char *material;
} sfem_problem_t;

static int SFEM_DEBUG_DUMP = 0;

int ISOLVER_EXPORT isolver_function_destroy_array(const isolver_function_t *info, void *ptr) {
    SFEM_UNUSED(info);
    free(ptr);
    return ISOLVER_FUNCTION_SUCCESS;
}

int ISOLVER_EXPORT isolver_function_create_array(const isolver_function_t *info,
                                                 size_t size,
                                                 void **ptr) {
    SFEM_UNUSED(info);

    *ptr = malloc(size);
    memset(*ptr, 0, size);
    return ISOLVER_FUNCTION_SUCCESS;
}

int ISOLVER_EXPORT isolver_function_init(isolver_function_t *info) {
    const char *SFEM_MESH_DIR = "[error] undefined";
    SFEM_READ_ENV(SFEM_MESH_DIR, );

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

    const char *SFEM_OUTPUT_DIR = "./sfem_output";
    SFEM_READ_ENV(SFEM_OUTPUT_DIR, );

    real_t SFEM_VISCOSITY = 1;
    real_t SFEM_MASS_DENSITY = 1;

    SFEM_READ_ENV(SFEM_VISCOSITY, atof);
    SFEM_READ_ENV(SFEM_MASS_DENSITY, atof);

    const char *SFEM_MATERIAL = "";
    SFEM_READ_ENV(SFEM_MATERIAL, );

    // const char *SFEM_INITIAL_GUESS = 0;
    // SFEM_READ_ENV(SFEM_INITIAL_GUESS, );

    SFEM_READ_ENV(SFEM_DEBUG_DUMP, atoi);

    printf(
        "sfem:\n"
        "- SFEM_DIRICHLET_NODESET=%s\n"
        "- SFEM_DIRICHLET_VALUE=%s\n"
        "- SFEM_DIRICHLET_COMPONENT=%s\n"
        "- SFEM_NEUMANN_SIDESET=%s\n"
        "- SFEM_NEUMANN_VALUE=%s\n"
        "- SFEM_NEUMANN_COMPONENT=%s\n"
        "- SFEM_MATERIAL=%s\n"
        "- SFEM_VISCOSITY=%g\n"
        "- SFEM_MASS_DENSITY=%g\n"
        "- SFEM_OUTPUT_DIR=%s\n"
        "- SFEM_DEBUG_DUMP=%d\n",
        SFEM_DIRICHLET_NODESET,
        SFEM_DIRICHLET_VALUE,
        SFEM_DIRICHLET_COMPONENT,
        SFEM_NEUMANN_SIDESET,
        SFEM_NEUMANN_VALUE,
        SFEM_NEUMANN_COMPONENT,
        SFEM_MATERIAL,
        SFEM_VISCOSITY,
        SFEM_MASS_DENSITY,
        SFEM_OUTPUT_DIR,
        SFEM_DEBUG_DUMP);

    if (!SFEM_MESH_DIR || !SFEM_DIRICHLET_NODESET) {
        return ISOLVER_FUNCTION_FAILURE;
    }

    struct stat st = {0};
    if (stat(SFEM_OUTPUT_DIR, &st) == -1) {
        mkdir(SFEM_OUTPUT_DIR, 0700);
    }

    mesh_t *mesh = (mesh_t *)malloc(sizeof(mesh_t));

    if (mesh_read(info->comm, SFEM_MESH_DIR, mesh)) {
        return ISOLVER_FUNCTION_FAILURE;
    }

    sfem_problem_t *problem = (sfem_problem_t *)malloc(sizeof(sfem_problem_t));

    read_dirichlet_conditions(mesh,
                              SFEM_DIRICHLET_NODESET,
                              SFEM_DIRICHLET_VALUE,
                              SFEM_DIRICHLET_COMPONENT,
                              &problem->dirichlet_conditions,
                              &problem->n_dirichlet_conditions);

    read_neumann_conditions(mesh,
                            SFEM_NEUMANN_SIDESET,
                            SFEM_NEUMANN_VALUE,
                            SFEM_NEUMANN_COMPONENT,
                            &problem->neumann_conditions,
                            &problem->n_neumann_conditions);

    // Redistribute Dirichlet nodes
    problem->mesh = mesh;
    problem->block_size = elem_manifold_dim(mesh->element_type) + 1;
    problem->output_dir = SFEM_OUTPUT_DIR;

    problem->rho = SFEM_MASS_DENSITY;
    problem->mu = SFEM_VISCOSITY;
    problem->material = SFEM_MATERIAL;
    problem->n2n_rowptr = NULL;
    problem->n2n_colidx = NULL;

    // Store problem
    info->private_data = (void *)problem;
    return ISOLVER_FUNCTION_SUCCESS;
}

int ISOLVER_EXPORT isolver_function_initial_guess(isolver_function_t *info,
                                                  isolver_scalar_t *const x) {
    sfem_problem_t *problem = (sfem_problem_t *)info->private_data;
    assert(problem);
    mesh_t *mesh = problem->mesh;
    assert(mesh);

    const char *SFEM_INITIAL_GUESS = 0;
    SFEM_READ_ENV(SFEM_INITIAL_GUESS, );

    if (SFEM_INITIAL_GUESS) {
        // FIXME
        if (array_read(info->comm,
                       SFEM_INITIAL_GUESS,
                       SFEM_MPI_REAL_T,
                       (void *)x,
                       mesh->nnodes,
                       mesh->nnodes)) {
            return ISOLVER_FUNCTION_FAILURE;
        }
    } else {
        memset(x, 0, mesh->nnodes * problem->block_size * sizeof(isolver_scalar_t));
    }

    return ISOLVER_FUNCTION_SUCCESS;
}

int ISOLVER_EXPORT isolver_function_create_crs_graph(const isolver_function_t *info,
                                                     ptrdiff_t *nlocal,
                                                     ptrdiff_t *nglobal,
                                                     ptrdiff_t *nnz,
                                                     isolver_idx_t **rowptr,
                                                     isolver_idx_t **colidx) {
    sfem_problem_t *problem = (sfem_problem_t *)info->private_data;
    assert(problem);
    mesh_t *mesh = problem->mesh;
    assert(mesh);

    if (!problem->n2n_rowptr) {
        // We create it only the first time
        build_crs_graph_for_elem_type(mesh->element_type,
                                      mesh->nelements,
                                      mesh->nnodes,
                                      mesh->elements,
                                      &problem->n2n_rowptr,
                                      &problem->n2n_colidx);
    }

    *rowptr = (count_t *)malloc((mesh->nnodes + 1) * problem->block_size * sizeof(count_t));
    *colidx = (idx_t *)malloc(problem->n2n_rowptr[mesh->nnodes] * problem->block_size *
                              problem->block_size * sizeof(idx_t));

    crs_graph_block_to_scalar(mesh->nnodes,
                              problem->block_size,
                              problem->n2n_rowptr,
                              problem->n2n_colidx,
                              *rowptr,
                              *colidx);

    *nlocal = mesh->nnodes * problem->block_size;
    *nglobal = mesh->nnodes * problem->block_size;
    *nnz = problem->n2n_rowptr[mesh->nnodes] * (problem->block_size * problem->block_size);

    return ISOLVER_FUNCTION_SUCCESS;
}

int ISOLVER_EXPORT isolver_function_create_vector(const isolver_function_t *info,
                                                  ptrdiff_t *nlocal,
                                                  ptrdiff_t *nglobal,
                                                  isolver_scalar_t **values) {
    sfem_problem_t *problem = (sfem_problem_t *)info->private_data;
    assert(problem);
    mesh_t *mesh = problem->mesh;
    assert(mesh);

    const size_t nbytes = mesh->nnodes * problem->block_size * sizeof(isolver_scalar_t);

    *values = (isolver_scalar_t *)malloc(nbytes);
    memset(*values, 0, nbytes);
    *nlocal = mesh->nnodes * problem->block_size;
    *nglobal = mesh->nnodes * problem->block_size;

    return ISOLVER_FUNCTION_SUCCESS;
}

int ISOLVER_EXPORT isolver_function_destroy_vector(const isolver_function_t *info,
                                                   isolver_scalar_t *values) {
    SFEM_UNUSED(info);

    free(values);
    return ISOLVER_FUNCTION_SUCCESS;
}

// Optimization function
int ISOLVER_EXPORT isolver_function_value(const isolver_function_t *info,
                                          const isolver_scalar_t *x,
                                          isolver_scalar_t *const out) {
    sfem_problem_t *problem = (sfem_problem_t *)info->private_data;
    assert(problem);
    mesh_t *mesh = problem->mesh;
    assert(mesh);

    // stokes_mini_assemble_value_aos(mesh->element_type,
    //                                  mesh->nelements,
    //                                  mesh->nnodes,
    //                                  mesh->elements,
    //                                  mesh->points,
    //                                  problem->mu,
    //                                  problem->rho,
    //                                  x,
    //                                  out);

    *out = 1;

    return ISOLVER_FUNCTION_SUCCESS;
}

int ISOLVER_EXPORT isolver_function_gradient(const isolver_function_t *info,
                                             const isolver_scalar_t *const x,
                                             isolver_scalar_t *const out) {
    sfem_problem_t *problem = (sfem_problem_t *)info->private_data;
    assert(problem);
    mesh_t *mesh = problem->mesh;
    assert(mesh);

    stokes_mini_assemble_gradient_aos(mesh->element_type,
                                      mesh->nelements,
                                      mesh->nnodes,
                                      mesh->elements,
                                      mesh->points,
                                      problem->mu,
                                      x,
                                      out);

    add_neumann_condition_to_gradient_vec(
        problem->n_neumann_conditions, problem->neumann_conditions, mesh, problem->block_size, out);

    if (SFEM_DEBUG_DUMP) {
        static int gradient_counter = 0;
        char path[1024 * 10];
        sprintf(path, "%s/g_debug_%d.raw", problem->output_dir, gradient_counter++);
        array_write(info->comm,
                    path,
                    SFEM_MPI_REAL_T,
                    out,
                    mesh->nnodes * problem->block_size,
                    mesh->nnodes * problem->block_size);
    }

    return ISOLVER_FUNCTION_SUCCESS;
}

// We might want to have also other formats here
int ISOLVER_EXPORT isolver_function_hessian_crs(const isolver_function_t *info,
                                                const isolver_scalar_t *const x,
                                                const isolver_idx_t *const rowptr,
                                                const isolver_idx_t *const colidx,
                                                isolver_scalar_t *const values) {
    SFEM_UNUSED(x);

    sfem_problem_t *problem = (sfem_problem_t *)info->private_data;
    assert(problem);
    mesh_t *mesh = problem->mesh;
    assert(mesh);

    stokes_mini_assemble_hessian_aos(mesh->element_type,
                                     mesh->nelements,
                                     mesh->nnodes,
                                     mesh->elements,
                                     mesh->points,
                                     problem->mu,
                                     problem->n2n_rowptr,
                                     problem->n2n_colidx,
                                     values);

    apply_dirichlet_condition_to_hessian_crs_vec(problem->n_dirichlet_conditions,
                                                 problem->dirichlet_conditions,
                                                 mesh,
                                                 problem->block_size,
                                                 rowptr,
                                                 colidx,
                                                 values);

    if (SFEM_DEBUG_DUMP) {
        static int hessian_counter = 0;

        crs_t crs_out;
        crs_out.rowptr = (char *)rowptr;
        crs_out.colidx = (char *)colidx;
        crs_out.values = (char *)values;
        crs_out.grows = mesh->nnodes * problem->block_size;
        crs_out.lrows = mesh->nnodes * problem->block_size;
        crs_out.lnnz = rowptr[mesh->nnodes * problem->block_size];
        crs_out.gnnz = rowptr[mesh->nnodes * problem->block_size];
        crs_out.start = 0;
        crs_out.rowoffset = 0;
        crs_out.rowptr_type = SFEM_MPI_COUNT_T;
        crs_out.colidx_type = SFEM_MPI_IDX_T;
        crs_out.values_type = SFEM_MPI_REAL_T;

        char path[1024 * 10];
        sprintf(path, "%s/H_debug_%d.raw", problem->output_dir, hessian_counter++);

        struct stat st = {0};
        if (stat(path, &st) == -1) {
            mkdir(path, 0700);
        }

        crs_write_folder(info->comm, path, &crs_out);
    }

    if (0) {
        for (ptrdiff_t i = 0; i < (mesh->nnodes * problem->block_size); i++) {
            printf("%ld\t", i);

            for (count_t k = rowptr[i]; k < rowptr[i + 1]; k++) {
                idx_t col = colidx[k];
                real_t val = values[k];

                printf("(%d, %g) ", col, val);
                assert(col < mesh->nnodes * problem->block_size);
            }
            printf("\n");
        }
    }

    return ISOLVER_FUNCTION_SUCCESS;
}

// Operator
int ISOLVER_EXPORT isolver_function_apply(const isolver_function_t *info,
                                          const isolver_scalar_t *const x,
                                          const isolver_scalar_t *const h,
                                          isolver_scalar_t *const out) {
    SFEM_UNUSED(x);

    sfem_problem_t *problem = (sfem_problem_t *)info->private_data;
    assert(problem);
    mesh_t *mesh = problem->mesh;
    assert(mesh);

    stokes_mini_apply_aos(mesh->element_type,
                          mesh->nelements,
                          mesh->nnodes,
                          mesh->elements,
                          mesh->points,
                          problem->mu,
                          h,
                          out);

    return ISOLVER_FUNCTION_SUCCESS;
}

int ISOLVER_EXPORT isolver_function_apply_constraints(const isolver_function_t *info,
                                                      isolver_scalar_t *const x) {
    sfem_problem_t *problem = (sfem_problem_t *)info->private_data;
    assert(problem);
    mesh_t *mesh = problem->mesh;
    assert(mesh);

    apply_dirichlet_condition_vec(problem->n_dirichlet_conditions,
                                  problem->dirichlet_conditions,
                                  mesh,
                                  problem->block_size,
                                  x);

    return ISOLVER_FUNCTION_SUCCESS;
}

int ISOLVER_EXPORT isolver_function_apply_zero_constraints(const isolver_function_t *info,
                                                           isolver_scalar_t *const x) {
    sfem_problem_t *problem = (sfem_problem_t *)info->private_data;
    assert(problem);
    mesh_t *mesh = problem->mesh;
    assert(mesh);

    apply_zero_dirichlet_condition_vec(problem->n_dirichlet_conditions,
                                       problem->dirichlet_conditions,
                                       mesh,
                                       problem->block_size,
                                       x);

    return ISOLVER_FUNCTION_SUCCESS;
}

int ISOLVER_EXPORT isolver_function_copy_constrained_dofs(const isolver_function_t *info,
                                                          const isolver_scalar_t *const src,
                                                          isolver_scalar_t *const dest) {
    sfem_problem_t *problem = (sfem_problem_t *)info->private_data;
    assert(problem);
    mesh_t *mesh = problem->mesh;
    assert(mesh);

    copy_at_dirichlet_nodes_vec(problem->n_dirichlet_conditions,
                                problem->dirichlet_conditions,
                                mesh,
                                problem->block_size,
                                src,
                                dest);

    return ISOLVER_FUNCTION_SUCCESS;
}

int ISOLVER_EXPORT isolver_function_report_solution(const isolver_function_t *info,
                                                    const isolver_scalar_t *const x) {
    sfem_problem_t *problem = (sfem_problem_t *)info->private_data;
    assert(problem);
    mesh_t *mesh = problem->mesh;
    assert(mesh);

    char path[2048];
    sprintf(path, "%s/out.raw", problem->output_dir);

    if (array_write(info->comm,
                    path,
                    SFEM_MPI_REAL_T,
                    x,
                    mesh->nnodes * problem->block_size,
                    mesh->nnodes * problem->block_size)) {
        return ISOLVER_FUNCTION_FAILURE;
    }

    return ISOLVER_FUNCTION_SUCCESS;
}

int ISOLVER_EXPORT isolver_function_destroy(isolver_function_t *info) {
    sfem_problem_t *problem = (sfem_problem_t *)info->private_data;
    free(problem->mesh);
    free(problem->n2n_rowptr);
    free(problem->n2n_colidx);

    destroy_conditions(problem->n_dirichlet_conditions, problem->dirichlet_conditions);
    destroy_conditions(problem->n_neumann_conditions, problem->neumann_conditions);

    return ISOLVER_FUNCTION_SUCCESS;
}
