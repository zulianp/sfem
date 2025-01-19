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
            fprintf(stderr, "usage: %s <tet15_mesh> <output_hex8_mesh>\n", argv[0]);
        }

        return EXIT_FAILURE;
    }

    const char     *output_folder  = argv[2];
    auto            tet15_elements = sfem::mesh_connectivity_from_file(comm, argv[1]);
    const ptrdiff_t n_elements     = tet15_elements->extent(1);
    auto            hex8_elements  = sfem::create_host_buffer<idx_t>(8, n_elements * 4);
    auto            elems          = hex8_elements->data();

    auto tet15_elems = tet15_elements->data();
    for (ptrdiff_t e = 0; e < n_elements; e++) {
        idx_t ii[15];
        for (ptrdiff_t d = 0; d < 15; d++) {
            ii[d] = tet15_elems[d][e];
        }

        idx_t hex8[4][8] = {// HEX8(0)
                            {ii[0], ii[4], ii[13], ii[6], ii[7], ii[10], ii[14], ii[12]},
                            // HEX8(1)
                            {ii[4], ii[1], ii[5], ii[13], ii[10], ii[8], ii[11], ii[14]},
                            // HEX8(2)
                            {ii[13], ii[5], ii[2], ii[6], ii[14], ii[11], ii[9], ii[12]},
                            // HEX8(3)
                            {ii[7], ii[10], ii[14], ii[12], ii[3], ii[8], ii[11], ii[9]}};

        for (int sub_e = 0; sub_e < 4; sub_e++) {
            for (int node = 0; node < 8; node++) {
                elems[node][ e * 4 + sub_e] = hex8[sub_e][node];
            }
        }
    }

    // Output
    sfem::create_directory(output_folder);

    std::string path_output_format = output_folder;
    path_output_format += "/i%d.raw";
    hex8_elements->to_files(path_output_format.c_str());

    // path_output_format = output_folder;
    // path_output_format += "/x%d.raw";
    // hex8_points->to_files(path_output_format.c_str());

    return MPI_Finalize();
}
