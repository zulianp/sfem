#include "sfem_API.hpp"

void create_ring_mesh(const geom_t    inner_radius,
                      const geom_t    outer_radius,
                      const ptrdiff_t nlayers,
                      const ptrdiff_t nelements,
                      idx_t** const   elements,
                      geom_t** const  points) {
    ptrdiff_t nnodes = nelements * 2;

    const geom_t dangle = 2 * M_PI / nelements;
   const geom_t dh     = (outer_radius - inner_radius) / nlayers;

    for (ptrdiff_t l = 0; l <= nlayers; l++) {
        for (ptrdiff_t i = 0; i < nelements; i++) {
            ptrdiff_t idx = l * nelements + i;
            points[0][idx] = cos(dangle * i) * (inner_radius + dh * l);
            points[1][idx] =  sin(dangle * i) * (inner_radius + dh * l);
        }
    }

    for (ptrdiff_t l = 0; l < nlayers; l++) {
        for (ptrdiff_t i = 0; i < nelements; i++) {
            ptrdiff_t idx = l * nelements + i;
            elements[0][idx] = l * nelements + (i + 1) % nelements;
            elements[1][idx] = l * nelements + i;
            elements[2][idx] = elements[1][idx] + nelements;
            elements[3][idx] = elements[0][idx] + nelements;
        }
    }
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    MPI_Comm comm = MPI_COMM_WORLD;

    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    if (argc != 6) {
        if (!rank) {
            fprintf(stderr, "usage: %s <inner_radius> <outer_radius> <nlayers> <nelements> <output_folder>\n", argv[0]);
        }

        return EXIT_FAILURE;
    }

    const geom_t    inner_radius  = atof(argv[1]);
    const geom_t    outer_radius  = atof(argv[2]);
    const ptrdiff_t nlayers       = atol(argv[3]);
    const ptrdiff_t nelements     = atol(argv[4]);
    std::string     output_folder = argv[5];

    double tick = MPI_Wtime();

    auto elements = sfem::create_host_buffer<idx_t>(4, nlayers * nelements);
    auto points   = sfem::create_host_buffer<geom_t>(3, (nlayers + 1) * nelements);

    create_ring_mesh(inner_radius, outer_radius, nlayers, nelements, elements->data(), points->data());

    // elements->print(std::cout);
    // points->print(std::cout);

    sfem::create_directory(output_folder.c_str());
    elements->to_files((output_folder + "/i%d.raw").c_str());
    points->to_files((output_folder + "/x%d.raw").c_str());

    double tock = MPI_Wtime();

    if (!rank) {
        printf("----------------------------------------\n");
        printf("TTS:\t\t\t%g seconds\n", tock - tick);
    }

    return MPI_Finalize();
}
