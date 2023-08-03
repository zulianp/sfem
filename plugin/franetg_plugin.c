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

#include "phase_field_for_fracture.h"

#include "read_mesh.h"
#include "sfem_defs.h"
#include "sfem_mesh.h"

#include "dirichlet.h"
#include "neumann.h"

typedef struct {
    ptrdiff_t local_size, global_size;
    idx_t *idx;
    int component;
    real_t value;
} boundary_condition_t;

typedef struct {
    mesh_t *mesh;

    count_t *n2n_rowptr;
    idx_t *n2n_colidx;
    int block_size;

    int n_neumann_conditions;
    boundary_condition_t *neumann_conditions;

    int n_dirichlet_conditions;
    boundary_condition_t *dirichlet_conditions;

    real_t mu, lambda, Gc, ls;

    const char *output_dir;
} sfem_problem_t;

void read_boundary_conditions(MPI_Comm comm,
                              char *sets,
                              char *values,
                              char *components,
                              boundary_condition_t **bcs,
                              int *nbc) {
    if (!sets) {
        *bcs = NULL;
        *nbc = 0;
        return;
    }

    const char *splitters = " ,;| ";
    char *pch;
    pch = strtok(sets, splitters);

    int count = 1;
    while (pch != NULL) {
        pch = strtok(NULL, splitters);
        count++;
    }

    boundary_condition_t *conds = malloc(count * sizeof(boundary_condition_t));

    // NODESET/SIDESET
    {
        pch = strtok(sets, splitters);
        int i = 0;
        while (pch != NULL) {
            printf("Reading %s\n", pch);
            pch = strtok(NULL, splitters);

            if (array_create_from_file(comm,
                                       pch,
                                       SFEM_MPI_IDX_T,
                                       (void **)&conds[i].idx,
                                       &conds[i].local_size,
                                       &conds[i].global_size)) {
                fprintf(stderr, "Failed to read file %s\n", pch);
                return;
            }

            // Some default values
            conds[i].value = 0;
            conds[i].component = 0;
            i++;
        }
    }

    if (values) {
        pch = strtok(values, splitters);
        int i = 0;
        while (pch != NULL) {
            printf("Reading %s\n", pch);
            pch = strtok(NULL, splitters);

            conds[i].value = atof(pch);
            i++;
        }
    }

    if (components) {
        pch = strtok(components, splitters);
        int i = 0;
        while (pch != NULL) {
            printf("Reading %s\n", pch);
            pch = strtok(NULL, splitters);

            conds[i].component = atoi(pch);
            i++;
        }
    }

    *bcs = conds;
    *nbc = count;
}

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

    // const char *SFEM_INITIAL_GUESS = 0;
    // SFEM_READ_ENV(SFEM_INITIAL_GUESS, );

    printf(
        "sfem:\n"
        "- SFEM_DIRICHLET_NODESET=%s\n"
        "- SFEM_DIRICHLET_VALUE=%s\n"
        "- SFEM_DIRICHLET_COMPONENT=%s\n"
        "- SFEM_NEUMANN_SIDESET=%s\n"
        "- SFEM_NEUMANN_VALUE=%s\n"
        "- SFEM_NEUMANN_COMPONENT=%s\n"
        "- SFEM_OUTPUT_DIR=%s\n",
        SFEM_DIRICHLET_NODESET,
        SFEM_DIRICHLET_VALUE,
        SFEM_DIRICHLET_COMPONENT,
        SFEM_NEUMANN_SIDESET,
        SFEM_NEUMANN_VALUE,
        SFEM_NEUMANN_COMPONENT,
        SFEM_OUTPUT_DIR);

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

    read_boundary_conditions(info->comm,
                             SFEM_DIRICHLET_NODESET,
                             SFEM_DIRICHLET_VALUE,
                             SFEM_DIRICHLET_COMPONENT,
                             &problem->dirichlet_conditions,
                             &problem->n_dirichlet_conditions);

    read_boundary_conditions(info->comm,
                             SFEM_NEUMANN_SIDESET,
                             SFEM_NEUMANN_VALUE,
                             SFEM_NEUMANN_COMPONENT,
                             &problem->neumann_conditions,
                             &problem->n_neumann_conditions);

    // Count faces not nodes
    enum ElemType stype = side_type(mesh->element_type);
    int nns = elem_num_sides(stype);

    for (int i = 0; i < problem->n_neumann_conditions; i++) {
        problem->neumann_conditions[i].global_size /= nns;
        problem->neumann_conditions[i].local_size /= nns;
    }

    // Redistribute Dirichlet nodes
    problem->mesh = mesh;
    problem->block_size = mesh->spatial_dim + 1;
    problem->output_dir = SFEM_OUTPUT_DIR;

    // Store problem
    info->private_data = (void *)problem;

    // Eventually initialize info->private_data here
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

    build_crs_graph(
        mesh->nelements, mesh->nnodes, mesh->elements, &problem->n2n_rowptr, &problem->n2n_colidx);

    *rowptr = (count_t *)malloc((mesh->nnodes + 1) * problem->block_size * sizeof(count_t));
    *colidx = (idx_t *)malloc(problem->n2n_rowptr[mesh->nnodes] * problem->block_size *
                              problem->block_size * sizeof(idx_t));

    crs_graph_block_to_scalar(mesh->nnodes,
                              problem->block_size,
                              problem->n2n_rowptr,
                              problem->n2n_colidx,
                              *rowptr,
                              *colidx);

    *nlocal = mesh->nnodes;
    *nglobal = mesh->nnodes;
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

    phase_field_for_fracture_assemble_value_aos(mesh->element_type,
                                                mesh->nelements,
                                                mesh->nnodes,
                                                mesh->elements,
                                                mesh->points,
                                                problem->mu,
                                                problem->lambda,
                                                problem->Gc,
                                                problem->ls,
                                                x,
                                                out);
    return ISOLVER_FUNCTION_SUCCESS;
}

int ISOLVER_EXPORT isolver_function_gradient(const isolver_function_t *info,
                                             const isolver_scalar_t *const x,
                                             isolver_scalar_t *const out) {
    sfem_problem_t *problem = (sfem_problem_t *)info->private_data;
    assert(problem);
    mesh_t *mesh = problem->mesh;
    assert(mesh);

    phase_field_for_fracture_assemble_gradient_aos(mesh->element_type,
                                                   mesh->nelements,
                                                   mesh->nnodes,
                                                   mesh->elements,
                                                   mesh->points,
                                                   problem->mu,
                                                   problem->lambda,
                                                   problem->Gc,
                                                   problem->ls,
                                                   x,
                                                   out);

    for (int i = 0; i < problem->n_neumann_conditions; i++) {
        surface_forcing_function_vec(
            // side_type(mesh->element_type), //FIXME
            problem->neumann_conditions[i].local_size,
            problem->neumann_conditions[i].idx,
            mesh->points,
            problem->neumann_conditions[i].value,
            problem->block_size,
            problem->neumann_conditions[i].component,
            out);
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

    phase_field_for_fracture_assemble_hessian_aos(mesh->element_type,
                                                  mesh->nelements,
                                                  mesh->nnodes,
                                                  mesh->elements,
                                                  mesh->points,
                                                  problem->mu,
                                                  problem->lambda,
                                                  problem->Gc,
                                                  problem->ls,
                                                  x,
                                                  problem->n2n_rowptr,
                                                  problem->n2n_colidx,
                                                  values);

    for (int i = 0; i < problem->n_dirichlet_conditions; i++) {
        crs_constraint_nodes_to_identity_vec(problem->dirichlet_conditions[i].local_size,
                                             problem->dirichlet_conditions[i].idx,
                                             problem->block_size,
                                             problem->dirichlet_conditions[i].component,
                                             problem->dirichlet_conditions[i].value,
                                             rowptr,
                                             colidx,
                                             values);
    }
    return ISOLVER_FUNCTION_SUCCESS;
}

// Operator
// int ISOLVER_EXPORT isolver_function_apply(const isolver_function_t *info,
//                                           const isolver_scalar_t *const x,
//                                           const isolver_scalar_t *const h,
//                                           isolver_scalar_t *const out) {
//     sfem_problem_t *problem = (sfem_problem_t *)info->private_data;
//     assert(problem);
//     mesh_t *mesh = problem->mesh;
//     assert(mesh);

//     // Equivalent to operator application due to linearity of the problem
//     // phase_field_for_fracture_assemble_gradient_aos(mesh->element_type,
//     //                                                mesh->nelements,
//     //                                                mesh->nnodes,
//     //                                                mesh->elements,
//     //                                                mesh->points,
//     //                                                h,
//     //                                                out);
//     assert(0);
//     return ISOLVER_FUNCTION_SUCCESS;
// }

int ISOLVER_EXPORT isolver_function_apply_constraints(const isolver_function_t *info,
                                                      isolver_scalar_t *const x) {
    sfem_problem_t *problem = (sfem_problem_t *)info->private_data;
    assert(problem);

    for (int i = 0; i < problem->n_dirichlet_conditions; i++) {
        constraint_nodes_to_value_vec(problem->dirichlet_conditions[i].local_size,
                                      problem->dirichlet_conditions[i].idx,
                                      problem->dirichlet_conditions[i].value,
                                      problem->block_size,
                                      problem->dirichlet_conditions[i].component,
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

    // No constraints for this example
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

    free(problem->dirichlet_conditions);

    if (problem->neumann_conditions) {
        free(problem->neumann_conditions);
    }

    return ISOLVER_FUNCTION_SUCCESS;
}
