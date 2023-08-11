#include "isolver_function.h"

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#include "matrix.io/array_dtof.h"
#include "matrix.io/matrixio_array.h"
#include "matrix.io/matrixio_crs.h"
#include "matrix.io/utils.h"

#include "crs_graph.h"
#include "sfem_base.h"

#include "linear_elasticity.h"
#include "neohookean.h"

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

    real_t mu, lambda;

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

    real_t SFEM_SHEAR_MODULUS = 2.23;
    real_t SFEM_FIRST_LAME_PARAMETER = 3.35;
    real_t SFEM_FRACTURE_TOUGHNESS = 0.27;
    real_t SFEM_LENGTH_SCALE = 1.;

    SFEM_READ_ENV(SFEM_SHEAR_MODULUS, atof);
    SFEM_READ_ENV(SFEM_FIRST_LAME_PARAMETER, atof);

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
        "- SFEM_SHEAR_MODULUS=%g\n"
        "- SFEM_FIRST_LAME_PARAMETER=%g\n"
        "- SFEM_OUTPUT_DIR=%s\n"
        "- SFEM_DEBUG_DUMP=%d\n",
        SFEM_DIRICHLET_NODESET,
        SFEM_DIRICHLET_VALUE,
        SFEM_DIRICHLET_COMPONENT,
        SFEM_NEUMANN_SIDESET,
        SFEM_NEUMANN_VALUE,
        SFEM_NEUMANN_COMPONENT,
        SFEM_MATERIAL,
        SFEM_SHEAR_MODULUS,
        SFEM_FIRST_LAME_PARAMETER,
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
    problem->block_size = elem_manifold_dim(mesh->element_type);
    problem->output_dir = SFEM_OUTPUT_DIR;

    problem->lambda = SFEM_FIRST_LAME_PARAMETER;
    problem->mu = SFEM_SHEAR_MODULUS;
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

    build_crs_graph_for_elem_type(mesh->element_type,
                                  mesh->nelements,
                                  mesh->nnodes,
                                  mesh->elements,
                                  &problem->n2n_rowptr,
                                  &problem->n2n_colidx);

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

    if (strcmp(problem->material, "linear") == 0) {
        linear_elasticity_assemble_value_aos(mesh->element_type,
                                             mesh->nelements,
                                             mesh->nnodes,
                                             mesh->elements,
                                             mesh->points,
                                             problem->mu,
                                             problem->lambda,
                                             x,
                                             out);
    } else {
        neohookean_assemble_value(  // mesh->element_type,
            mesh->nelements,
            mesh->nnodes,
            mesh->elements,
            mesh->points,
            problem->mu,
            problem->lambda,
            x,
            out);
    }
    return ISOLVER_FUNCTION_SUCCESS;
}

int ISOLVER_EXPORT isolver_function_gradient(const isolver_function_t *info,
                                             const isolver_scalar_t *const x,
                                             isolver_scalar_t *const out) {
    sfem_problem_t *problem = (sfem_problem_t *)info->private_data;
    assert(problem);
    mesh_t *mesh = problem->mesh;
    assert(mesh);

    if (strcmp(problem->material, "linear") == 0) {
        linear_elasticity_assemble_gradient_aos(mesh->element_type,
                                                mesh->nelements,
                                                mesh->nnodes,
                                                mesh->elements,
                                                mesh->points,
                                                problem->mu,
                                                problem->lambda,
                                                x,
                                                out);
    } else {
        neohookean_assemble_gradient(  // mesh->element_type,
            mesh->nelements,
            mesh->nnodes,
            mesh->elements,
            mesh->points,
            problem->mu,
            problem->lambda,
            x,
            out);
    }

    for (int i = 0; i < problem->n_neumann_conditions; i++) {
        surface_forcing_function_vec(side_type(mesh->element_type),
                                     problem->neumann_conditions[i].local_size,
                                     problem->neumann_conditions[i].idx,
                                     mesh->points,
                                     -  // Use negative sign since we are on LHS
                                     problem->neumann_conditions[i].value,
                                     problem->block_size,
                                     problem->neumann_conditions[i].component,
                                     out);
    }

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
    sfem_problem_t *problem = (sfem_problem_t *)info->private_data;
    assert(problem);
    mesh_t *mesh = problem->mesh;
    assert(mesh);

    if (strcmp(problem->material, "linear") == 0) {
        linear_elasticity_assemble_hessian_aos(mesh->element_type,
                                               mesh->nelements,
                                               mesh->nnodes,
                                               mesh->elements,
                                               mesh->points,
                                               problem->mu,
                                               problem->lambda,
                                               problem->n2n_rowptr,
                                               problem->n2n_colidx,
                                               values);
    } else {
        neohookean_assemble_hessian(  // mesh->element_type,
            mesh->nelements,
            mesh->nnodes,
            mesh->elements,
            mesh->points,
            problem->mu,
            problem->lambda,
            x,
            problem->n2n_rowptr,
            problem->n2n_colidx,
            values);
    }

    for (int i = 0; i < problem->n_dirichlet_conditions; i++) {
        crs_constraint_nodes_to_identity_vec(problem->dirichlet_conditions[i].local_size,
                                             problem->dirichlet_conditions[i].idx,
                                             problem->block_size,
                                             problem->dirichlet_conditions[i].component,
                                             1,
                                             rowptr,
                                             colidx,
                                             values);
    }

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
    sfem_problem_t *problem = (sfem_problem_t *)info->private_data;
    assert(problem);
    mesh_t *mesh = problem->mesh;
    assert(mesh);

    if (strcmp(problem->material, "linear") == 0) {
        linear_elasticity_apply_aos(mesh->element_type,
                                    mesh->nelements,
                                    mesh->nnodes,
                                    mesh->elements,
                                    mesh->points,
                                    problem->mu,
                                    problem->lambda,
                                    h,
                                    out);
    } else {
        // TODO
        assert(0);
    }

    return ISOLVER_FUNCTION_SUCCESS;
}

int ISOLVER_EXPORT isolver_function_apply_constraints(const isolver_function_t *info,
                                                      isolver_scalar_t *const x) {
    sfem_problem_t *problem = (sfem_problem_t *)info->private_data;
    assert(problem);

    for (int i = 0; i < problem->n_dirichlet_conditions; i++) {
        constraint_nodes_to_value_vec(problem->dirichlet_conditions[i].local_size,
                                      problem->dirichlet_conditions[i].idx,
                                      problem->block_size,
                                      problem->dirichlet_conditions[i].component,
                                      problem->dirichlet_conditions[i].value,
                                      x);
    }

    return ISOLVER_FUNCTION_SUCCESS;
}

int ISOLVER_EXPORT isolver_function_apply_zero_constraints(const isolver_function_t *info,
                                                           isolver_scalar_t *const x) {
    sfem_problem_t *problem = (sfem_problem_t *)info->private_data;
    assert(problem);

    for (int i = 0; i < problem->n_dirichlet_conditions; i++) {
        constraint_nodes_to_value_vec(problem->dirichlet_conditions[i].local_size,
                                      problem->dirichlet_conditions[i].idx,
                                      problem->block_size,
                                      problem->dirichlet_conditions[i].component,
                                      0,
                                      x);
    }

    return ISOLVER_FUNCTION_SUCCESS;
}

int ISOLVER_EXPORT isolver_function_copy_constrained_dofs(const isolver_function_t *info,
                                                          const isolver_scalar_t *const src,
                                                          isolver_scalar_t *const dest) {
    sfem_problem_t *problem = (sfem_problem_t *)info->private_data;
    assert(problem);
    mesh_t *mesh = problem->mesh;
    assert(mesh);

    for (int i = 0; i < problem->n_dirichlet_conditions; i++) {
        constraint_nodes_copy_vec(problem->dirichlet_conditions[i].local_size,
                                  problem->dirichlet_conditions[i].idx,
                                  problem->block_size,
                                  problem->dirichlet_conditions[i].component,
                                  src,
                                  dest);
    }

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

    for (int i = 0; i < problem->n_dirichlet_conditions; i++) {
        free(problem->dirichlet_conditions[i].idx);
    }

    free(problem->dirichlet_conditions);

    if (problem->neumann_conditions) {
        for (int i = 0; i < problem->n_neumann_conditions; i++) {
            free(problem->neumann_conditions[i].idx);
        }

        free(problem->neumann_conditions);
    }

    return ISOLVER_FUNCTION_SUCCESS;
}
