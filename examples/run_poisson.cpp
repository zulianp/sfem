#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include "sfem_base.h"
#include "sfem_defs.h"


#include "boundary_condition.h"
#include "dirichlet.h"
#include "neumann.h"

#include "matrixio_array.h"

#include "laplacian.h"
#include "read_mesh.h"
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

    // struct stat st = {0};
    // if (stat(output_folder, &st) == -1) {
    //     mkdir(output_folder, 0700);
    // }

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

    if (rank == 0) {
        printf(
            "----------------------------------------\n"
            "Options:\n"
            "----------------------------------------\n"
            "- SFEM_DIRICHLET_NODESET=%s\n"
            "- SFEM_DIRICHLET_VALUE=%s\n"
            "- SFEM_DIRICHLET_COMPONENT=%s\n"
            "----------------------------------------\n",
            SFEM_DIRICHLET_NODESET,
            SFEM_DIRICHLET_VALUE,
            SFEM_DIRICHLET_COMPONENT);
    }

    int n_dirichlet_conditions;
    boundary_condition_t *dirichlet_conditions;
    read_dirichlet_conditions(&mesh,
                              SFEM_DIRICHLET_NODESET,
                              SFEM_DIRICHLET_VALUE,
                              SFEM_DIRICHLET_COMPONENT,
                              &dirichlet_conditions,
                              &n_dirichlet_conditions);

    const int sdim = elem_manifold_dim((ElemType)mesh.element_type);

    sfem::ConjugateGradient<real_t> cg;
    cg.max_it = 1000;
    cg.default_init();
    cg.apply_op = [&](const real_t *const x, real_t *const y) {
        memset(y, 0, mesh.nnodes * sizeof(real_t));
        laplacian_apply((ElemType)mesh.element_type,
                        mesh.nelements,
                        mesh.nnodes,
                        mesh.elements,
                        mesh.points,
                        x,
                        y);

        copy_at_dirichlet_nodes_vec(n_dirichlet_conditions, dirichlet_conditions, &mesh, 1, x, y);
    };

    std::vector<real_t> x(mesh.nnodes, 0), b(mesh.nnodes, 0);

    apply_dirichlet_condition_vec(n_dirichlet_conditions, dirichlet_conditions, &mesh, 1, x.data());
    apply_dirichlet_condition_vec(n_dirichlet_conditions, dirichlet_conditions, &mesh, 1, b.data());

    cg.apply(mesh.nnodes, b.data(), x.data());

    array_write(comm, output_path, SFEM_MPI_REAL_T, x.data(), mesh.nnodes, mesh.nnodes);

    ptrdiff_t nelements = mesh.nelements;
    ptrdiff_t nnodes = mesh.nnodes;

    mesh_destroy(&mesh);
    destroy_conditions(n_dirichlet_conditions, dirichlet_conditions);
    // destroy_conditions(n_neumann_conditions, neumann_conditions);

    double tock = MPI_Wtime();
    if (!rank) {
        printf("----------------------------------------\n");
        printf("#elements %ld #nodes %ld\n", (long)nelements, (long)nnodes);
        printf("TTS:\t\t\t%g seconds\n", tock - tick);
    }

    return MPI_Finalize();
}
