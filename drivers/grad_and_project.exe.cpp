#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "array_dtof.h"
#include "matrixio_array.h"
#include "matrixio_crs.h"
#include "utils.h"

#include "crs_graph.h"
#include "sfem_base.h"

#include "sfem_defs.h"

#include "operators/div.h"
#include "tet4_grad.h"

#include "operators/tet4/tet4_l2_projection_p0_p1.h"

#include "operators/tet10/tet10_grad.h"
#include "operators/tet10/tet10_l2_projection_p1_p2.h"

#include "read_mesh.h"

#include "sfem_API.hpp"

void tet4_p1_p1_grad_and_project(const ptrdiff_t nelements,
                                 const ptrdiff_t nnodes,
                                 idx_t **const SFEM_RESTRICT elems,
                                 geom_t **const SFEM_RESTRICT xyz,
                                 const real_t *const SFEM_RESTRICT u,
                                 real_t *const SFEM_RESTRICT dudx,
                                 real_t *const SFEM_RESTRICT dudy,
                                 real_t *const SFEM_RESTRICT dudz)

{
    real_t *p0_dudx = (real_t*)malloc(nelements * sizeof(real_t));
    real_t *p0_dudy = (real_t*)malloc(nelements * sizeof(real_t));
    real_t *p0_dudz = (real_t*)malloc(nelements * sizeof(real_t));

    tet4_grad(nelements, nnodes, elems, xyz, u, p0_dudx, p0_dudy, p0_dudz);

    tet4_p0_p1_l2_projection_apply(nelements, nnodes, elems, xyz, p0_dudx, dudx);
    tet4_p0_p1_l2_projection_apply(nelements, nnodes, elems, xyz, p0_dudy, dudy);
    tet4_p0_p1_l2_projection_apply(nelements, nnodes, elems, xyz, p0_dudz, dudz);

    free(p0_dudx);
    free(p0_dudy);
    free(p0_dudz);
}

// Should this routine use "mass-lumping" for the projection?
void tet10_p2_p2_grad_and_project(const ptrdiff_t nelements,
                                  const ptrdiff_t nnodes,
                                  idx_t **const SFEM_RESTRICT elems,
                                  geom_t **const SFEM_RESTRICT xyz,
                                  const real_t *const SFEM_RESTRICT u,
                                  real_t *const SFEM_RESTRICT dudx,
                                  real_t *const SFEM_RESTRICT dudy,
                                  real_t *const SFEM_RESTRICT dudz)

{
    real_t *p1_dudx = (real_t*)malloc(nelements * 4 * sizeof(real_t));
    real_t *p1_dudy = (real_t*)malloc(nelements * 4 * sizeof(real_t));
    real_t *p1_dudz = (real_t*)malloc(nelements * 4 * sizeof(real_t));

    tet10_grad(nelements, nnodes, elems, xyz, u, p1_dudx, p1_dudy, p1_dudz);

    tet10_ep1_p2_l2_projection_apply(nelements, nnodes, elems, xyz, p1_dudx, dudx);
    tet10_ep1_p2_l2_projection_apply(nelements, nnodes, elems, xyz, p1_dudy, dudy);
    tet10_ep1_p2_l2_projection_apply(nelements, nnodes, elems, xyz, p1_dudz, dudz);

    free(p1_dudx);
    free(p1_dudy);
    free(p1_dudz);
}

void grad_and_project(const enum ElemType element_type,
                      const ptrdiff_t nelements,
                      const ptrdiff_t nnodes,
                      idx_t **const SFEM_RESTRICT elems,
                      geom_t **const SFEM_RESTRICT xyz,
                      const real_t *const SFEM_RESTRICT u,
                      real_t *const SFEM_RESTRICT dudx,
                      real_t *const SFEM_RESTRICT dudy,
                      real_t *const SFEM_RESTRICT dudz)

{
    switch (element_type) {
        case TET4: {
            tet4_p1_p1_grad_and_project(nelements, nnodes, elems, xyz, u, dudx, dudy, dudz);
            break;
        }
        case TET10: {
            tet10_p2_p2_grad_and_project(nelements, nnodes, elems, xyz, u, dudx, dudy, dudz);
            break;
        }
        default: {
            assert(0);
            MPI_Abort(MPI_COMM_WORLD, SFEM_FAILURE);
        }
    }
}

//

void tet4_p1_p1_grad_and_project_coeffs(const ptrdiff_t nelements,
                                        const ptrdiff_t nnodes,
                                        idx_t **const SFEM_RESTRICT elems,
                                        geom_t **const SFEM_RESTRICT xyz,
                                        const real_t *const SFEM_RESTRICT u,
                                        real_t *const SFEM_RESTRICT dudx,
                                        real_t *const SFEM_RESTRICT dudy,
                                        real_t *const SFEM_RESTRICT dudz)

{
    real_t *p0_dudx = (real_t*)malloc(nelements * sizeof(real_t));
    real_t *p0_dudy = (real_t*)malloc(nelements * sizeof(real_t));
    real_t *p0_dudz = (real_t*)malloc(nelements * sizeof(real_t));

    tet4_grad(nelements, nnodes, elems, xyz, u, p0_dudx, p0_dudy, p0_dudz);

    tet4_p0_p1_projection_coeffs(nelements, nnodes, elems, xyz, p0_dudx, dudx);
    tet4_p0_p1_projection_coeffs(nelements, nnodes, elems, xyz, p0_dudy, dudy);
    tet4_p0_p1_projection_coeffs(nelements, nnodes, elems, xyz, p0_dudz, dudz);

    free(p0_dudx);
    free(p0_dudy);
    free(p0_dudz);
}

void print_if_not_zero(const ptrdiff_t nelements, real_t *p1_dudx)
{
    for (ptrdiff_t e = 0; e < nelements; e++) {
        int printed = 0;
        for (int i = 0; i < 4; i++) {
            if (fabs(p1_dudx[e * 4 + i]) > 1e-4) {
                printf("%3.2lf ", p1_dudx[e * 4 + i]);
                printed = 1;
            }
        }

        if (printed) {
            printf("\n");
        }
    }

}

// Should this routine use "mass-lumping" for the projection?
void tet10_p2_p2_grad_and_project_coeffs(const ptrdiff_t nelements,
                                         const ptrdiff_t nnodes,
                                         idx_t **const SFEM_RESTRICT elems,
                                         geom_t **const SFEM_RESTRICT xyz,
                                         const real_t *const SFEM_RESTRICT u,
                                         real_t *const SFEM_RESTRICT dudx,
                                         real_t *const SFEM_RESTRICT dudy,
                                         real_t *const SFEM_RESTRICT dudz)

{
    real_t *p1_dudx = (real_t*)malloc(nelements * 4 * sizeof(real_t));
    real_t *p1_dudy = (real_t*)malloc(nelements * 4 * sizeof(real_t));
    real_t *p1_dudz = (real_t*)malloc(nelements * 4 * sizeof(real_t));

    tet10_grad(nelements, nnodes, elems, xyz, u, p1_dudx, p1_dudy, p1_dudz);

    // print_if_not_zero(nelements, p1_dudx);
    // print_if_not_zero(nelements, p1_dudy);
    // print_if_not_zero(nelements, p1_dudz);

    tet10_ep1_p2_projection_coeffs(nelements, nnodes, elems, xyz, p1_dudx, dudx);
    tet10_ep1_p2_projection_coeffs(nelements, nnodes, elems, xyz, p1_dudy, dudy);
    tet10_ep1_p2_projection_coeffs(nelements, nnodes, elems, xyz, p1_dudz, dudz);

    free(p1_dudx);
    free(p1_dudy);
    free(p1_dudz);
}

void grad_and_project_coeffs(const enum ElemType element_type,
                             const ptrdiff_t nelements,
                             const ptrdiff_t nnodes,
                             idx_t **const SFEM_RESTRICT elems,
                             geom_t **const SFEM_RESTRICT xyz,
                             const real_t *const SFEM_RESTRICT u,
                             real_t *const SFEM_RESTRICT dudx,
                             real_t *const SFEM_RESTRICT dudy,
                             real_t *const SFEM_RESTRICT dudz)

{
    switch (element_type) {
        case TET4: {
            tet4_p1_p1_grad_and_project_coeffs(nelements, nnodes, elems, xyz, u, dudx, dudy, dudz);
            break;
        }
        case TET10: {
            tet10_p2_p2_grad_and_project_coeffs(nelements, nnodes, elems, xyz, u, dudx, dudy, dudz);
            break;
        }
        default: {
            assert(0);
            MPI_Abort(MPI_COMM_WORLD, SFEM_FAILURE);
        }
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

    if (argc != 6) {
        fprintf(stderr, "usage: %s <folder> <u.raw> <dudx.raw> <dudy.raw> <dudz.raw>\n", argv[0]);
        return EXIT_FAILURE;
    }

    const char *folder = argv[1];
    const char *path_u = argv[2];
    const char *path_output[3] = {argv[3], argv[4], argv[5]};

    printf("%s %s %s %s %s %s\n", argv[0], folder, path_u, path_output[0], path_output[1], path_output[2]);

    double tick = MPI_Wtime();

    ///////////////////////////////////////////////////////////////////////////////
    // Read data
    ///////////////////////////////////////////////////////////////////////////////

    auto mesh = sfem::Mesh::create_from_file(comm, folder);
    const ptrdiff_t n_elements = mesh->n_elements();
    const ptrdiff_t n_nodes = mesh->n_nodes();

    real_t *u;

    ptrdiff_t u_n_local, u_n_global;
    array_create_from_file(comm, path_u, SFEM_MPI_REAL_T, (void **)&u, &u_n_local, &u_n_global);

    assert(u_n_local == n_nodes);

    real_t *grad_u[3];
    for (int d = 0; d < mesh->spatial_dimension(); ++d) {
        grad_u[d] = (real_t *)malloc(u_n_local * sizeof(real_t));
        memset(grad_u[d], 0, u_n_local * sizeof(real_t));
    }

    int SFEM_COMPUTE_COEFFICIENTS = 1;

    SFEM_READ_ENV(SFEM_COMPUTE_COEFFICIENTS, atoi);

    if (SFEM_COMPUTE_COEFFICIENTS) {
        grad_and_project_coeffs(mesh->element_type(),
                                n_elements,
                                n_nodes,
                                mesh->elements()->data(),
                                mesh->points()->data(),
                                u,
                                grad_u[0],
                                grad_u[1],
                                grad_u[2]);

    } else {
        grad_and_project(mesh->element_type(),
                         n_elements,
                         n_nodes,
                         mesh->elements()->data(),
                         mesh->points()->data(),
                         u,
                         grad_u[0],
                         grad_u[1],
                         grad_u[2]);
    }

    real_t SFEM_SCALE = 1;
    SFEM_READ_ENV(SFEM_SCALE, atof);

    if (SFEM_SCALE != 1) {
        for (int d = 0; d < mesh->spatial_dimension(); ++d) {
            for (ptrdiff_t i = 0; i < u_n_local; ++i) {
                grad_u[d][i] *= SFEM_SCALE;
            }
        }
    }

    free(u);

    for (int d = 0; d < mesh->spatial_dimension(); ++d) {
        array_write(comm, path_output[d], SFEM_MPI_REAL_T, grad_u[d], u_n_local, u_n_global);
        free(grad_u[d]);
    }

    double tock = MPI_Wtime();

    if (!rank) {
        printf("----------------------------------------\n");
        printf("#elements %ld #nodes %ld\n", (long)n_elements, (long)n_nodes);
        printf("TTS:\t\t\t%g seconds\n", tock - tick);
    }

    return MPI_Finalize();
}
