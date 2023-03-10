#include "isolver_function.h"

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "matrix.io/array_dtof.h"
#include "matrix.io/matrixio_array.h"
#include "matrix.io/matrixio_crs.h"
#include "matrix.io/utils.h"

#include "crs_graph.h"
#include "sfem_base.h"

#include "laplacian.h"

#include "read_mesh.h"
#include "sfem_mesh.h"

#include "dirichlet.h"
#include "neumann.h"

typedef struct {
    mesh_t *mesh;
    idx_t *dirichlet_nodes;
    ptrdiff_t nlocal_dirchlet;
    ptrdiff_t nglobal_dirchlet;

    ptrdiff_t nlocal_neumann;
    ptrdiff_t nglobal_neumann;
    idx_t *faces_neumann;

    int block_size;

    const char *output_dir;
} sfem_problem_t;

int ISOLVER_EXPORT isolver_function_destroy_array(const isolver_function_t *info, void *ptr) {
    free(ptr);
    return ISOLVER_FUNCTION_SUCCESS;
}

int ISOLVER_EXPORT isolver_function_create_array(const isolver_function_t *info, size_t size, void **ptr) {
    *ptr = malloc(size);
    memset(*ptr, 0, size);
    return ISOLVER_FUNCTION_SUCCESS;
}

int ISOLVER_EXPORT isolver_function_init(isolver_function_t *info) {
    const char *SFEM_MESH_DIR = "[error] undefined";
    const char *SFEM_MATERIAL_MODEL = "[error] undefined";
    const char *SFEM_DIRICHLET_NODES = "[error] undefined";
    const char *SFEM_OUTPUT_DIR = "./sfem_output";
    const char *SFEM_NEUMAN_FACES = "[error] undefined";

    SFEM_READ_ENV(SFEM_MESH_DIR, );
    SFEM_READ_ENV(SFEM_MATERIAL_MODEL, );
    SFEM_READ_ENV(SFEM_DIRICHLET_NODES, );
    SFEM_READ_ENV(SFEM_OUTPUT_DIR, );
    SFEM_READ_ENV(SFEM_NEUMAN_FACES, );

    printf(
        "sfem:\n"
        "- SFEM_MESH_DIR=%s\n"
        "- SFEM_MATERIAL_MODEL%s\n"
        "- SFEM_DIRICHLET_NODES=%s\n"
        "- SFEM_OUTPUT_DIR=%s\n"
        "- SFEM_NEUMAN_FACES=%s\n",
        SFEM_MESH_DIR,
        SFEM_MATERIAL_MODEL,
        SFEM_DIRICHLET_NODES,
        SFEM_OUTPUT_DIR,
        SFEM_NEUMAN_FACES);

    if (!SFEM_MESH_DIR || !SFEM_DIRICHLET_NODES || !SFEM_NEUMAN_FACES) {
        return ISOLVER_FUNCTION_FAILURE;
    }

    mesh_t *mesh = (mesh_t *)malloc(sizeof(mesh_t));

    if (mesh_read(info->comm, SFEM_MESH_DIR, mesh)) {
        return ISOLVER_FUNCTION_FAILURE;
    }

    sfem_problem_t *problem = (sfem_problem_t *)malloc(sizeof(sfem_problem_t));

    if (array_create_from_file(info->comm,
                   SFEM_DIRICHLET_NODES,
                   SFEM_MPI_IDX_T,
                   (void **)&problem->dirichlet_nodes,
                   &problem->nlocal_dirchlet,
                   &problem->nglobal_dirchlet)) {
        return ISOLVER_FUNCTION_FAILURE;
    }

    if (array_create_from_file(info->comm,
                   SFEM_NEUMAN_FACES,
                   SFEM_MPI_IDX_T,
                   (void **)&problem->faces_neumann,
                   &problem->nlocal_neumann,
                   &problem->nglobal_neumann)) {
        return ISOLVER_FUNCTION_FAILURE;
    }

    problem->nlocal_neumann /= 3;
    problem->nglobal_neumann /= 3;

    // Redistribute Dirichlet nodes
    problem->mesh = mesh;
    problem->block_size = 1;
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

    build_crs_graph(mesh->nelements, mesh->nnodes, mesh->elements, rowptr, colidx);

    *nlocal = mesh->nnodes;
    *nglobal = mesh->nnodes;
    *nnz = (*rowptr)[*nlocal] * (problem->block_size * problem->block_size);
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

    const size_t nbytes = mesh->nnodes * sizeof(isolver_scalar_t);

    *values = (isolver_scalar_t *)malloc(nbytes);
    memset(*values, 0, nbytes);
    *nlocal = mesh->nnodes;
    *nglobal = mesh->nnodes;

    return ISOLVER_FUNCTION_SUCCESS;
}

int ISOLVER_EXPORT isolver_function_destroy_vector(const isolver_function_t *info, isolver_scalar_t *values) {
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

    laplacian_assemble_value(mesh->nelements, mesh->nnodes, mesh->elements, mesh->points, x, out);
    return ISOLVER_FUNCTION_SUCCESS;
}

int ISOLVER_EXPORT isolver_function_gradient(const isolver_function_t *info,
                                             const isolver_scalar_t *const x,
                                             isolver_scalar_t *const out) {
    sfem_problem_t *problem = (sfem_problem_t *)info->private_data;
    assert(problem);
    mesh_t *mesh = problem->mesh;
    assert(mesh);

    laplacian_assemble_gradient(mesh->nelements, mesh->nnodes, mesh->elements, mesh->points, x, out);
    surface_forcing_function(problem->nlocal_neumann, problem->faces_neumann, mesh->points, -1, out);

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

    laplacian_assemble_hessian(mesh->nelements, mesh->nnodes, mesh->elements, mesh->points, rowptr, colidx, values);

    crs_constraint_nodes_to_identity(problem->nlocal_dirchlet, problem->dirichlet_nodes, 1.0, rowptr, colidx, values);
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

    // Equivalent to operator application due to linearity of the problem
    laplacian_assemble_gradient(mesh->nelements, mesh->nnodes, mesh->elements, mesh->points, h, out);
    return ISOLVER_FUNCTION_SUCCESS;
}

int ISOLVER_EXPORT isolver_function_apply_constraints(const isolver_function_t *info, isolver_scalar_t *const x) {
    sfem_problem_t *problem = (sfem_problem_t *)info->private_data;
    assert(problem);

    constraint_nodes_to_value(problem->nlocal_dirchlet, problem->dirichlet_nodes, 0, x);
    return ISOLVER_FUNCTION_SUCCESS;
}

int ISOLVER_EXPORT isolver_function_apply_zero_constraints(const isolver_function_t *info, isolver_scalar_t *const x) {
    sfem_problem_t *problem = (sfem_problem_t *)info->private_data;
    assert(problem);

    constraint_nodes_to_value(problem->nlocal_dirchlet, problem->dirichlet_nodes, 0, x);
    return ISOLVER_FUNCTION_SUCCESS;
}

int ISOLVER_EXPORT isolver_function_copy_constrained_dofs(const isolver_function_t *info,
                                                          const isolver_scalar_t *const src,
                                                          isolver_scalar_t *const dest) {
    sfem_problem_t *problem = (sfem_problem_t *)info->private_data;
    assert(problem);
    mesh_t *mesh = problem->mesh;
    assert(mesh);

    constraint_nodes_copy(problem->nlocal_dirchlet, problem->dirichlet_nodes, src, dest);

    // No constraints for this example
    return ISOLVER_FUNCTION_SUCCESS;
}

int ISOLVER_EXPORT isolver_function_report_solution(const isolver_function_t *info, const isolver_scalar_t *const x) {
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
    free(problem->dirichlet_nodes);
    free(problem->faces_neumann);
    return ISOLVER_FUNCTION_SUCCESS;
}
