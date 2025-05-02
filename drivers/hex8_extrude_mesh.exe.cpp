#include "sfem_API.hpp"

#include "sortreduce.h"

static SFEM_INLINE void normalize(real_t* const vec3) {
    const real_t len = sqrt(vec3[0] * vec3[0] + vec3[1] * vec3[1] + vec3[2] * vec3[2]);
    vec3[0] /= len;
    vec3[1] /= len;
    vec3[2] /= len;
}

static SFEM_INLINE void normal(const idx_t                  i0,
                               const idx_t                  i1,
                               const idx_t                  i2,
                               geom_t** const SFEM_RESTRICT xyz,
                               real_t* const SFEM_RESTRICT  n) {
    real_t u[3] = {xyz[0][i1] - xyz[0][i0], xyz[1][i1] - xyz[1][i0], xyz[2][i1] - xyz[2][i0]};
    real_t v[3] = {xyz[0][i2] - xyz[0][i0], xyz[1][i2] - xyz[1][i0], xyz[2][i2] - xyz[2][i0]};

    normalize(u);
    normalize(v);

    n[0] = u[1] * v[2] - u[2] * v[1];
    n[1] = u[2] * v[0] - u[0] * v[2];
    n[2] = u[0] * v[1] - u[1] * v[0];

    normalize(n);
}

void extrude(const ptrdiff_t              nsides,
             const ptrdiff_t              nnodes,
             idx_t** const SFEM_RESTRICT  sides,
             geom_t** const SFEM_RESTRICT xyz,
             const ptrdiff_t              nlayers,
             const geom_t                 height,
             idx_t** const SFEM_RESTRICT  extruded_elements,
             geom_t** const SFEM_RESTRICT extruded_xyz) {
    double tick = MPI_Wtime();

    geom_t** pseudo_normals = (geom_t**)malloc(3 * sizeof(geom_t*));
    for (int d = 0; d < 3; d++) {
        pseudo_normals[d] = (geom_t*)calloc(nnodes, sizeof(geom_t));
    }

    for (ptrdiff_t i = 0; i < nsides; ++i) {
        const idx_t i0 = sides[0][i];
        const idx_t i1 = sides[1][i];
        const idx_t i2 = sides[2][i];

        real_t n[3];
        normal(i0, i1, i2, xyz, n);

        for (int d = 0; d < 3; d++) {
            pseudo_normals[d][i0] += n[d];
            pseudo_normals[d][i1] += n[d];
            pseudo_normals[d][i2] += n[d];
        }
    }

    for (ptrdiff_t i = 0; i < nnodes; ++i) {
        real_t n[3];
        for (int d = 0; d < 3; d++) {
            n[d] = pseudo_normals[d][i];
        }

        normalize(n);

        for (int d = 0; d < 3; d++) {
            pseudo_normals[d][i] = n[d];
        }
    }

    for (ptrdiff_t l = 0; l < nlayers; l++) {
        // Construct hexx8 elements
        for (ptrdiff_t i = 0; i < nsides; ++i) {
            for (int d = 0; d < 4; d++) {
                idx_t node = sides[d][i];

                idx_t node_bottom = l * nnodes + node;
                idx_t node_top    = (l + 1) * nnodes + node;

                extruded_elements[d][l * nsides + i]     = node_bottom;
                extruded_elements[4 + d][l * nsides + i] = node_top;
            }
        }
    }

    const geom_t dh = height / nlayers;
    for (ptrdiff_t l = 0; l <= nlayers; l++) {
        for (ptrdiff_t i = 0; i < nnodes; ++i) {
            for (int dd = 0; dd < 3; dd++) {
                extruded_xyz[dd][l * nnodes + i] = xyz[dd][i] + (l * dh * pseudo_normals[dd][i]);
            }
        }
    }

    double tock = MPI_Wtime();
    printf("extrude.c: extrude\t%g seconds\n", tock - tick);
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    MPI_Comm comm = MPI_COMM_WORLD;

    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    if (argc != 5) {
        if (!rank) {
            fprintf(stderr, "usage: %s <quad4_mesh> <height> <nlayers> <output_hex8_mesh>\n", argv[0]);
        }

        return EXIT_FAILURE;
    }

    const char*     input_folder  = argv[1];
    const geom_t    height        = atof(argv[2]);
    const ptrdiff_t nlayers       = atol(argv[3]);
    const char*     output_folder = argv[4];

    auto quad_mesh = sfem::Mesh::create_from_file(comm, input_folder);

    auto hex8_elements = sfem::create_host_buffer<idx_t>(8, quad_mesh->n_elements() * nlayers);
    auto hex8_points   = sfem::create_host_buffer<geom_t>(3, quad_mesh->n_nodes() * (nlayers + 1));

    extrude(quad_mesh->n_elements(),
            quad_mesh->n_nodes(),
            quad_mesh->elements()->data(),
            quad_mesh->points()->data(),
            nlayers,
            height,
            hex8_elements->data(),
            hex8_points->data());

    sfem::create_directory(output_folder);
    
    std::string path_output_format = output_folder;
    path_output_format += "/i%d.raw";
    hex8_elements->to_files(path_output_format.c_str());

    path_output_format = output_folder;
    path_output_format += "/x%d.raw";
    hex8_points->to_files(path_output_format.c_str());

    return MPI_Finalize();
}
