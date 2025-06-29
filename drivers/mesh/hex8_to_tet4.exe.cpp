#include "sfem_API.hpp"

#include "sortreduce.h"

#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    MPI_Comm comm = MPI_COMM_WORLD;

    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    if (argc != 3) {
        if (!rank) {
            fprintf(stderr, "usage: %s <hex8_mesh> <tet4_mesh>\n", argv[0]);
        }

        return EXIT_FAILURE;
    }

    const char     *output_folder = argv[2];
    auto            hex8_elements = sfem::mesh_connectivity_from_file(sfem::Communicator::wrap(comm), argv[1]);
    const ptrdiff_t n_elements    = hex8_elements->extent(1);
    auto            tet4_elements = sfem::create_host_buffer<idx_t>(4, n_elements * 6);
    auto            elems         = tet4_elements->data();

    auto hex8_elems = hex8_elements->data();
    for (ptrdiff_t e = 0; e < n_elements; e++) {
        idx_t ii[8];
        for (ptrdiff_t d = 0; d < 8; d++) {
            ii[d] = hex8_elems[d][e];
        }

        idx_t hex8[6][4] = {{ii[0], ii[1], ii[3], ii[7]},
                            {ii[0], ii[1], ii[7], ii[5]},
                            {ii[0], ii[4], ii[5], ii[7]},
                            {ii[1], ii[2], ii[3], ii[6]},
                            {ii[1], ii[3], ii[7], ii[6]},
                            {ii[1], ii[5], ii[6], ii[7]}};

        for (int sub_e = 0; sub_e < 6; sub_e++) {
            for (int node = 0; node < 4; node++) {
                elems[node][e * 6 + sub_e] = hex8[sub_e][node];
            }
        }
    }

    // Output
    sfem::create_directory(output_folder);

    std::string path_output_format = output_folder;
    path_output_format += "/i%d.raw";
    tet4_elements->to_files(path_output_format.c_str());
    return MPI_Finalize();
}
