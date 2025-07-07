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
                               geom_t** const SFEM_RESTRICT points,
                               real_t* const SFEM_RESTRICT  n) {
    real_t u[3] = {points[0][i1] - points[0][i0], points[1][i1] - points[1][i0], points[2][i1] - points[2][i0]};
    real_t v[3] = {points[0][i2] - points[0][i0], points[1][i2] - points[1][i0], points[2][i2] - points[2][i0]};

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
             geom_t** const SFEM_RESTRICT points,
             const ptrdiff_t              nlayers,
             const geom_t                 height,
             idx_t** const SFEM_RESTRICT  extruded_elements,
             geom_t** const SFEM_RESTRICT extruded_points) {
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
        normal(i0, i1, i2, points, n);

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
                extruded_points[dd][l * nnodes + i] = points[dd][i] + (l * dh * pseudo_normals[dd][i]);
            }
        }
    }

    double tock = MPI_Wtime();
    printf("extrude.c: extrude\t%g seconds\n", tock - tick);
}

void        translate3(const ptrdiff_t             nnodes,
                       const geom_t                tx,
                       const geom_t                ty,
                       const geom_t                tz,
                       geom_t* const SFEM_RESTRICT x,
                       geom_t* const SFEM_RESTRICT y,
                       geom_t* const SFEM_RESTRICT z) {
    if(tx == 0 && ty == 0 && tz == 0) return;

#pragma omp parallel for
    for (ptrdiff_t i = 0; i < nnodes; i++) {
        x[i] += tx;
        y[i] += ty;
        z[i] += tz;
    }
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

    auto quad_mesh = sfem::Mesh::create_from_file(sfem::Communicator::wrap(comm), input_folder);

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

    geom_t SFEM_TRANSLATE_X = 0;
    geom_t SFEM_TRANSLATE_Y = 0;
    geom_t SFEM_TRANSLATE_Z = 0;

    SFEM_READ_ENV(SFEM_TRANSLATE_X, atof);
    SFEM_READ_ENV(SFEM_TRANSLATE_Y, atof);
    SFEM_READ_ENV(SFEM_TRANSLATE_Z, atof);

    translate3(hex8_points->extent(1),
               SFEM_TRANSLATE_X,
               SFEM_TRANSLATE_Y,
               SFEM_TRANSLATE_Z,
               hex8_points->data()[0],
               hex8_points->data()[1],
               hex8_points->data()[2]);

    sfem::create_directory(output_folder);

    std::string path_output_format = output_folder;
    path_output_format += "/i%d.raw";
    hex8_elements->to_files(path_output_format.c_str());

    path_output_format = output_folder;
    path_output_format += "/x%d.raw";
    hex8_points->to_files(path_output_format.c_str());

    return MPI_Finalize();
}
