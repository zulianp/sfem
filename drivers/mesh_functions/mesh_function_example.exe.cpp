#include "sfem_API.hpp"

void eval_function(const ptrdiff_t nnodes, geom_t **points, real_t *f) {
    SFEM_TRACE_SCOPE("eval_function");
    
#pragma omp parallel for
    for (ptrdiff_t i = 0; i < nnodes; i++) {
        geom_t x = points[0][i];
        geom_t y = points[1][i];
        geom_t z = points[2][i];
        f[i]     = x * y * z;
    }
}

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    MPI_Comm comm = MPI_COMM_WORLD;

    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    if (argc != 3) {
        if (!rank) {
            fprintf(stderr, "usage: %s <mesh> <function.raw>\n", argv[0]);
        }

        return EXIT_FAILURE;
    }

    auto        mesh   = sfem::Mesh::create_from_file(comm, argv[1]);
    const char *output = argv[2];

    const ptrdiff_t n_elements = mesh->n_elements();
    const ptrdiff_t n_nodes    = mesh->n_nodes();

    auto points = mesh->points()->data();
    auto f      = sfem::create_host_buffer<real_t>(n_nodes);
    eval_function(n_nodes, points, f->data());
    f->to_file(output);

    return MPI_Finalize();
}
