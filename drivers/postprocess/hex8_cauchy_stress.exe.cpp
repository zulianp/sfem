#include "sfem_API.hpp"

#include "hex8_linear_elasticity.h"
#include "hex8_mass.h"

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    MPI_Comm comm = MPI_COMM_WORLD;

    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    if (argc != 8) {
        if (!rank) {
            fprintf(stderr, "usage: %s <hex8_mesh> <mu> <lambda> <ux> <uy> <uz> <output_prefix>\n", argv[0]);
        }

        return EXIT_FAILURE;
    }

    auto        hex8_mesh     = sfem::Mesh::create_from_file(comm, argv[1]);
    real_t      mu            = atof(argv[2]);
    real_t      lambda        = atof(argv[3]);
    auto        ux            = sfem::create_buffer_from_file<real_t>(comm, argv[4]);
    auto        uy            = sfem::create_buffer_from_file<real_t>(comm, argv[5]);
    auto        uz            = sfem::create_buffer_from_file<real_t>(comm, argv[6]);
    const char *output_prefix = argv[7];

    const ptrdiff_t nnodes = hex8_mesh->n_nodes();

    auto stress = sfem::create_host_buffer<real_t>(6, nnodes);
    auto mass   = sfem::create_host_buffer<real_t>(nnodes);

    hex8_linear_elasticity_l2_project_cauchy_stress(hex8_mesh->n_elements(),
                                                    nnodes,
                                                    hex8_mesh->elements()->data(),
                                                    hex8_mesh->points()->data(),
                                                    mu,
                                                    lambda,
                                                    1,
                                                    ux->data(),
                                                    uy->data(),
                                                    uz->data(),
                                                    1,
                                                    stress->data()[0],
                                                    stress->data()[1],
                                                    stress->data()[2],
                                                    stress->data()[3],
                                                    stress->data()[4],
                                                    stress->data()[5]);

    hex8_assemble_lumped_mass(
            hex8_mesh->n_elements(), nnodes, hex8_mesh->elements()->data(), hex8_mesh->points()->data(), 1, mass->data());

    auto s = stress->data();
    auto m = mass->data();

#pragma omp parallel for
    for (int k = 0; k < 6; k++) {
        for (ptrdiff_t i = 0; i < nnodes; i++) {
            assert(s[k][i] == s[k][i]);
            assert(m[i] == m[i]);

            s[k][i] /= m[i];
        }
    }

    std::string path_output_format = output_prefix;
    path_output_format += ".%d.raw";
    stress->to_files(path_output_format.c_str());
    return MPI_Finalize();
}
