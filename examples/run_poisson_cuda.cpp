#include <math.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include "boundary_condition.h"
#include "sfem_base.h"
#include "sfem_defs.h"

#include "boundary_condition_incore_cuda.h"
#include "dirichlet.h"
#include "neumann.h"

#include "matrixio_array.h"

#include "laplacian.h"

#include "macro_tet4_laplacian_incore_cuda.h"
#include "tet10_laplacian_incore_cuda.h"
#include "tet4_laplacian_incore_cuda.h"

#include "read_mesh.h"
#include "sfem_bcgs.hpp"
#include "sfem_cg.hpp"

#include <vector>

#include "sfem_cuda_blas.h"

template <typename T>
void cg_init_cuda(sfem::ConjugateGradient<T> &cg) {
    cg.allocate = d_allocate;
    cg.destroy = d_destroy;
    cg.copy = d_copy;
    cg.dot = d_dot;
    cg.axpby = d_axpby;
    // cg.zaxpby =
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

    if (argc != 3) {
        fprintf(stderr, "usage: %s <folder> <output>\n", argv[0]);
        return EXIT_FAILURE;
    }

    const char *output_path = argv[2];

    double tick = MPI_Wtime();

    ///////////////////////////////////////////////////////////////////////////////
    // Read data
    ///////////////////////////////////////////////////////////////////////////////

    const char *folder = argv[1];

    mesh_t mesh;
    if (mesh_read(comm, folder, &mesh)) {
        return EXIT_FAILURE;
    }

    char *SFEM_DIRICHLET_NODESET = 0;
    char *SFEM_DIRICHLET_VALUE = 0;
    char *SFEM_DIRICHLET_COMPONENT = 0;
    SFEM_READ_ENV(SFEM_DIRICHLET_NODESET, );
    SFEM_READ_ENV(SFEM_DIRICHLET_VALUE, );
    SFEM_READ_ENV(SFEM_DIRICHLET_COMPONENT, );

    char *SFEM_NEUMANN_SIDESET = 0;
    char *SFEM_NEUMANN_VALUE = 0;
    SFEM_READ_ENV(SFEM_NEUMANN_SIDESET, );
    SFEM_READ_ENV(SFEM_NEUMANN_VALUE, );

    int SFEM_USE_PRECONDITIONER = 0;
    SFEM_READ_ENV(SFEM_USE_PRECONDITIONER, atoi);

    int SFEM_USE_MACRO = 1;
    SFEM_READ_ENV(SFEM_USE_MACRO, atoi);

    if (rank == 0) {
        printf(
            "----------------------------------------\n"
            "Options:\n"
            "----------------------------------------\n"
            "- SFEM_DIRICHLET_NODESET=%s\n"
            "- SFEM_DIRICHLET_VALUE=%s\n"
            "- SFEM_DIRICHLET_COMPONENT=%s\n"
            "- SFEM_USE_PRECONDITIONER=%d\n"
            "----------------------------------------\n",
            SFEM_DIRICHLET_NODESET,
            SFEM_DIRICHLET_VALUE,
            SFEM_DIRICHLET_COMPONENT,
            SFEM_USE_PRECONDITIONER);
    }

    int n_dirichlet_conditions;
    boundary_condition_t *d_dirichlet_conditions = 0;

    {   
        // 
        boundary_condition_t *h_dirichlet_conditions;
        read_dirichlet_conditions(&mesh,
                                  SFEM_DIRICHLET_NODESET,
                                  SFEM_DIRICHLET_VALUE,
                                  SFEM_DIRICHLET_COMPONENT,
                                  &h_dirichlet_conditions,
                                  &n_dirichlet_conditions);

        d_dirichlet_conditions =
            (boundary_condition_t *)malloc(n_dirichlet_conditions * sizeof(boundary_condition_t));

        for (int d = 0; d < n_dirichlet_conditions; d++) {
            boundary_conditions_host_to_device(&h_dirichlet_conditions[d],
                                               &d_dirichlet_conditions[d]);
        }

        destroy_conditions(n_dirichlet_conditions, h_dirichlet_conditions);
    }

    enum ElemType elem_type = (ElemType)mesh.element_type;

    if (SFEM_USE_MACRO) {
        elem_type = macro_type_variant(elem_type);
    }

    using Solver_t = sfem::ConjugateGradient<real_t>;
    // using Solver_t = sfem::BiCGStab<real_t>;

    Solver_t solver;

    solver.max_it = 9000;
    solver.tol = 1e-10;
    // solver.default_init();
    cg_init_cuda(solver);

    real_t *d_x = d_allocate(mesh.nnodes);
    real_t *d_b = d_allocate(mesh.nnodes);

    cuda_incore_laplacian_t ctx;

    if (mesh.element_type == TET4) {
        tet4_cuda_incore_laplacian_init(&ctx, mesh.nelements, mesh.elements, mesh.points);
    } else if (mesh.element_type == TET10) {
        // Go for macro just for testing
        if (SFEM_USE_MACRO) {
            macro_tet4_cuda_incore_laplacian_init(&ctx, mesh.nelements, mesh.elements, mesh.points);
        } else {
            tet10_cuda_incore_laplacian_init(&ctx, mesh.nelements, mesh.elements, mesh.points);
        }
    }

    solver.apply_op = [&](const real_t *const x, real_t *const y) {
        d_memset(y, 0, mesh.nnodes * sizeof(real_t));

        if (mesh.element_type == TET4) {
            tet4_cuda_incore_laplacian_apply(&ctx, x, y);
        } else if (mesh.element_type == TET10) {
            if (SFEM_USE_MACRO) {
                macro_tet4_cuda_incore_laplacian_apply(&ctx, x, y);
            } else {
                tet10_cuda_incore_laplacian_apply(&ctx, x, y);
            }
        }

        d_copy_at_dirichlet_nodes_vec(n_dirichlet_conditions, d_dirichlet_conditions, 1, x, y);
    };

    std::vector<real_t> x(mesh.nnodes, 0);

    d_apply_dirichlet_condition_vec(n_dirichlet_conditions, d_dirichlet_conditions, 1, d_x);
    d_apply_dirichlet_condition_vec(n_dirichlet_conditions, d_dirichlet_conditions, 1, d_b);

    solver.apply(mesh.nnodes, d_b, d_x);

    device_to_host(mesh.nnodes, d_x, x.data());

    array_write(comm, output_path, SFEM_MPI_REAL_T, x.data(), mesh.nnodes, mesh.nnodes);

    ptrdiff_t nelements = mesh.nelements;
    ptrdiff_t nnodes = mesh.nnodes;

    mesh_destroy(&mesh);
    d_destroy_conditions(n_dirichlet_conditions, d_dirichlet_conditions);
    // destroy_conditions(n_neumann_conditions, neumann_conditions);

    if (mesh.element_type == TET4) {
        tet4_cuda_incore_laplacian_destroy(&ctx);
    } else if (mesh.element_type == TET10) {
        if (SFEM_USE_MACRO) {
            macro_tet4_cuda_incore_laplacian_destroy(&ctx);
        } else {
            tet10_cuda_incore_laplacian_destroy(&ctx);
        }
    }

    d_destroy(d_b);
    d_destroy(d_x);

    double tock = MPI_Wtime();
    if (!rank) {
        printf("----------------------------------------\n");
        printf("#elements %ld #nodes %ld\n", (long)nelements, (long)nnodes);
        printf("TTS:\t\t\t%g seconds\n", tock - tick);
    }

    return MPI_Finalize();
}
