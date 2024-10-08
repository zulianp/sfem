#include <math.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include "sfem_base.h"
#include "sfem_defs.h"

#include "boundary_condition.h"
#include "boundary_condition_io.h"
#include "dirichlet.h"
#include "neumann.h"

#include "matrixio_array.h"

#include "laplacian.h"

#include "macro_tet4_laplacian.h"

#include "read_mesh.h"
#include "sfem_bcgs.hpp"
#include "sfem_cg.hpp"

#include <vector>

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

    int SFEM_USE_OPT = 1;
    int SFEM_USE_MACRO = 1;

    SFEM_READ_ENV(SFEM_USE_OPT, atoi);
    SFEM_READ_ENV(SFEM_USE_MACRO, atoi);

    if(SFEM_USE_OPT) {
        SFEM_USE_OPT = laplacian_is_opt(SFEM_USE_OPT);
    }

    if (rank == 0) {
        printf("----------------------------------------\n"
               "Options:\n"
               "----------------------------------------\n"
               "- SFEM_DIRICHLET_NODESET=%s\n"
               "- SFEM_DIRICHLET_VALUE=%s\n"
               "- SFEM_DIRICHLET_COMPONENT=%s\n"
               "- SFEM_USE_PRECONDITIONER=%d\n"
               "- SFEM_USE_OPT=%d\n"
               "- SFEM_USE_MACRO=%d\n"
               "----------------------------------------\n",
               SFEM_DIRICHLET_NODESET,
               SFEM_DIRICHLET_VALUE,
               SFEM_DIRICHLET_COMPONENT,
               SFEM_USE_PRECONDITIONER,
               SFEM_USE_OPT,
               SFEM_USE_MACRO);
    }

    int n_dirichlet_conditions;
    boundary_condition_t *dirichlet_conditions;
    read_dirichlet_conditions(&mesh,
                              SFEM_DIRICHLET_NODESET,
                              SFEM_DIRICHLET_VALUE,
                              SFEM_DIRICHLET_COMPONENT,
                              &dirichlet_conditions,
                              &n_dirichlet_conditions);

    enum ElemType elem_type = (ElemType)mesh.element_type;

    if (SFEM_USE_MACRO) {
        elem_type = macro_type_variant(elem_type);
    }

    // using Solver_t = sfem::ConjugateGradient<real_t>;
    using Solver_t = sfem::BiCGStab<real_t>;

    Solver_t solver;

    solver.max_it = 9000;
    solver.tol = 1e-10;
    solver.default_init();

    std::vector<real_t> diag;

    fff_t fff;
    if (SFEM_USE_OPT) {
        tet4_fff_create(&fff, mesh.nelements, mesh.elements, mesh.points);
    }

    if (SFEM_USE_PRECONDITIONER) {
        diag.resize(mesh.nnodes, 0);

        laplacian_diag(elem_type, mesh.nelements, mesh.nnodes, mesh.elements, mesh.points,diag.data());

        solver.set_preconditioner([&](const real_t *const x, real_t *const y) {

#pragma omp parallel for
            for (ptrdiff_t i = 0; i < mesh.nnodes; i++) {
                y[i] = x[i] / diag[i];
            }

            copy_at_dirichlet_nodes_vec(n_dirichlet_conditions, dirichlet_conditions, 1, x, y);
        });
    }

    solver.apply_op = [&](const real_t *const x, real_t *const y) {
        memset(y, 0, mesh.nnodes * sizeof(real_t));

        if (SFEM_USE_OPT) {
            laplacian_apply_opt(elem_type, fff.nelements, fff.elements, fff.data, x, y);
        } else {
            laplacian_apply(
                    elem_type, mesh.nelements, mesh.nnodes, mesh.elements, mesh.points, x, y);
        }

        copy_at_dirichlet_nodes_vec(n_dirichlet_conditions, dirichlet_conditions, 1, x, y);
    };

    std::vector<real_t> x(mesh.nnodes, 0), b(mesh.nnodes, 0);

    apply_dirichlet_condition_vec(n_dirichlet_conditions, dirichlet_conditions, 1, x.data());
    apply_dirichlet_condition_vec(n_dirichlet_conditions, dirichlet_conditions, 1, b.data());

    solver.apply(mesh.nnodes, b.data(), x.data());

    array_write(comm, output_path, SFEM_MPI_REAL_T, x.data(), mesh.nnodes, mesh.nnodes);

    ptrdiff_t nelements = mesh.nelements;
    ptrdiff_t nnodes = mesh.nnodes;

    mesh_destroy(&mesh);
    destroy_conditions(n_dirichlet_conditions, dirichlet_conditions);
    // destroy_conditions(n_neumann_conditions, neumann_conditions);

    if (elem_type == MACRO_TET4) {
        tet4_fff_destroy(&fff);
    }

    double tock = MPI_Wtime();
    if (!rank) {
        printf("----------------------------------------\n");
        printf("#elements %ld #nodes %ld\n", (long)nelements, (long)nnodes);
        printf("TTS:\t\t\t%g seconds\n", tock - tick);
    }

    return MPI_Finalize();
}
