#include <math.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include "boundary_condition.h"
#include "boundary_condition_io.h"
#include "sfem_base.h"
#include "sfem_defs.h"

#include "boundary_condition_incore_cuda.h"
#include "dirichlet.h"
#include "neumann.h"

#include "matrixio_array.h"

#include "laplacian.h"

#include "laplacian_incore_cuda.h"

#include "read_mesh.h"
#include "sfem_bcgs.hpp"
#include "sfem_cg.hpp"

#include <vector>

#include "sfem_cuda_blas.h"

template <typename T>
void sfem_cuda_init_solver(sfem::ConjugateGradient<T> &cg) {
    cg.allocate = d_allocate;
    cg.destroy = d_destroy;
    cg.copy = d_copy;
    cg.dot = d_dot;
    cg.axpby = d_axpby;
}

template <typename T>
void sfem_cuda_init_solver(sfem::BiCGStab<T> &cg) {
    cg.allocate = d_allocate;
    cg.destroy = d_destroy;
    cg.copy = d_copy;
    cg.dot = d_dot;
    cg.axpby = d_axpby;
    cg.zaxpby = d_zaxpby;
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

    // -------------------------------
    // Read inputs
    // -------------------------------

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

    // -------------------------------
    // Boundary conditions set-up
    // -------------------------------

    int n_dirichlet_conditions;
    boundary_condition_t *d_dirichlet_conditions = 0;

    {
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

    // -------------------------------
    // Solver set-up
    // -------------------------------

    // using Solver_t = sfem::ConjugateGradient<real_t>;
    using Solver_t = sfem::BiCGStab<real_t>;

    Solver_t solver;

    solver.max_it = 9000;
    solver.tol = 1e-10;
    // solver.default_init();
    sfem_cuda_init_solver(solver);

    real_t *d_x = d_allocate(mesh.nnodes);
    real_t *d_b = d_allocate(mesh.nnodes);
    real_t *d_d = nullptr;

    cuda_incore_laplacian_t ctx;
    cuda_incore_laplacian_init(elem_type, &ctx, mesh.nelements, mesh.elements, mesh.points);

    if (SFEM_USE_PRECONDITIONER) {
        d_d = d_allocate(mesh.nnodes);

        // Make sure preconditioner is available for element
        SFEM_USE_PRECONDITIONER = !cuda_incore_laplacian_diag(&ctx, d_d);

        if (SFEM_USE_PRECONDITIONER) {
            solver.set_preconditioner([&](const real_t *const x, real_t *const y) {
                d_ediv(mesh.nnodes, x, d_d, y);
                d_copy_at_dirichlet_nodes_vec(
                    n_dirichlet_conditions, d_dirichlet_conditions, 1, x, y);
            });
        } else {
            fprintf(stderr, "[Warning] preconditioner not avaialble for selected element\n");
            d_destroy(d_d);
            d_d = nullptr;
        }
    }

    solver.apply_op = [&](const real_t *const x, real_t *const y) {
        d_memset(y, 0, mesh.nnodes * sizeof(real_t));
        cuda_incore_laplacian_apply(&ctx, x, y);
        d_copy_at_dirichlet_nodes_vec(n_dirichlet_conditions, d_dirichlet_conditions, 1, x, y);
    };

    // -------------------------------
    // Solve
    // -------------------------------

    d_apply_dirichlet_condition_vec(n_dirichlet_conditions, d_dirichlet_conditions, 1, d_x);
    d_apply_dirichlet_condition_vec(n_dirichlet_conditions, d_dirichlet_conditions, 1, d_b);

    solver.apply(mesh.nnodes, d_b, d_x);

    // -------------------------------
    // Save to disk
    // -------------------------------

    std::vector<real_t> x(mesh.nnodes, 0);
    device_to_host(mesh.nnodes, d_x, x.data());
    array_write(comm, output_path, SFEM_MPI_REAL_T, x.data(), mesh.nnodes, mesh.nnodes);

    ptrdiff_t nelements = mesh.nelements;
    ptrdiff_t nnodes = mesh.nnodes;

    // -------------------------------
    // Clean-up
    // -------------------------------

    mesh_destroy(&mesh);
    d_destroy_conditions(n_dirichlet_conditions, d_dirichlet_conditions);
    // destroy_conditions(n_neumann_conditions, neumann_conditions);

    cuda_incore_laplacian_destroy(&ctx);

    d_destroy(d_b);
    d_destroy(d_x);

    if (d_d) {
        d_destroy(d_d);
    }

    // -------------------------------
    // Stats
    // -------------------------------

    double tock = MPI_Wtime();
    if (!rank) {
        printf("----------------------------------------\n");
        printf("#elements %ld #nodes %ld\n", (long)nelements, (long)nnodes);
        printf("TTS:\t\t\t%g seconds\n", tock - tick);
    }

    return MPI_Finalize();
}
